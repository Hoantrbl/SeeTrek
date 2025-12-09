from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from PIL import Image


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


SPATIAL_BASE = "/home/yijiexu/lipengteng/vsibench/tools/SeeTrek/modified_dataset_yolov8n_2hz_every_4_frames/"
SPATIAL_BASE_TSI = "/home/yijiexu/lipengteng/vsibench/tools/tsibench_yolov8n_2hz_every_4_frames/"

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def get_index_with_selection_by_count_diverse(
    counts, fps, max_frame, num_segments=8,
    bound=None, first_idx=0
):
    """
    从 counts = [[frame_num, [class_names]], ...] 中选择关键帧，优先选择全局最多物体的一帧，
    其余按照分段选取尽可能类别不同的关键帧。

    参数：
        counts: List[[int, List[str]]], 每项为 [frame_num, class_names]
        fps: float, 视频帧率
        max_frame: int, 视频最大帧数
        num_segments: int, 总共选择的关键帧数
        bound: Optional[List[float, float]], 时间范围
        first_idx: int, 起始帧索引下限

    返回：
        List[int], 所选关键帧帧号
    """
    # Step 1: 时间范围筛选
    start_time, end_time = bound if bound else (-1e9, 1e9)
    start_idx = max(first_idx, round(start_time * fps))
    end_idx = min(round(end_time * fps), max_frame)

    valid = [(f, names) for f, names in counts if start_idx <= f <= end_idx]
    if len(valid) < num_segments:
        raise ValueError("Not enough valid frames to select from.")

    valid.sort(key=lambda x: x[0])
    frame_nums = [f for f, _ in valid]
    object_counts = [len(names) for _, names in valid]
    class_names = [set(names) for _, names in valid]

    # Step 2: 全局选一个最多物体数（最早）帧
    global_best_idx = max(range(len(valid)), key=lambda i: (object_counts[i], -frame_nums[i]))
    selected_indices = [global_best_idx]
    selected_classes = class_names[global_best_idx]

    # Step 3: 分段选 num_segments - 1 个多样性帧
    total = len(valid)
    remaining_segments = num_segments - 1
    segment_len = total // remaining_segments if remaining_segments > 0 else total

    for i in range(remaining_segments):
        start = i * segment_len
        end = (i + 1) * segment_len if i < remaining_segments - 1 else total
        segment = list(range(start, end))
        if not segment:
            continue

        # 最少重叠类别 + 最多物体数 + 最早优先
        best_idx = min(
            segment,
            key=lambda idx: (
            len(class_names[idx] & selected_classes) - object_counts[idx]*2,  # 类别重叠少与物体数多评分1:2
            frame_nums[idx]                                                 # 越早越好
            )
        )
        selected_indices.append(best_idx)
        selected_classes.update(class_names[best_idx])

    # 最终返回帧号
    return np.array(sorted([frame_nums[i] for i in selected_indices]))

def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None

# 这里是入口，读取具体的帧数以及对应的video
def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    # nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    nframes = 8
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy() # 获取视频帧
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps

def _read_video_decord_spa(
    ele: dict,
    points: np.ndarray,
    frame_select_list: list,
    counts: list = None,
    num_segments: int = 8,
) -> (torch.Tensor, float, list):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(frame_select_list), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    # Uniformly sample indices from frame_select_list
    try:
        frame_indices = get_index_with_selection_by_count_diverse(counts, video_fps, total_frames, num_segments=8, bound=None, first_idx=0)
        if len(frame_indices) == 0:
            raise ValueError("Frame indices are empty after count-based sampling.")
    except ValueError as e:
        # Fall back to the original method
        frame_indices = np.linspace(0, len(frame_select_list) - 1, num_segments, dtype=int)
        frame_indices = [frame_select_list[idx] for idx in frame_indices]

    # frame_indices = np.linspace(0, len(frame_select_list) - 1, nframes, dtype=int)
    # frame_indices = [frame_select_list[idx] for idx in frame_indices]

    # Get the corresponding points information for each frame index
    idx_per_frame_indices = [frame_select_list.index(frame_index) for frame_index in frame_indices]
    points_info = [points[frame_select_list.index(frame_index)] for frame_index in frame_indices]

    frame_list = []
    num_points = points.shape[0]
    colors = (plt.cm.viridis(np.linspace(0, 1, num_points))[:, :3] * 255).astype(np.uint8)[:, ::-1]

    for i, frame_index in enumerate(frame_indices):
        frame = vr[frame_index].asnumpy()[:, :, ::-1].copy()  # RGB -> BGR, 直接拷贝为 OpenCV 格式

        color = tuple(int(c) for c in colors[idx_per_frame_indices[i]])  # 取预计算颜色
        center = (frame.shape[1] - 50, 50)
        
        cv2.circle(frame, center, 20, color, -1)

        text = str(frame_index)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # cv2.imwrite("test.png", frame.astype(np.uint8))
        # Draw the points on the image ===

        # Convert back to PIL image
        img = Image.fromarray(frame[:, :, ::-1])
        frame_list.append(img)

    scene_name = video_path.split("/")[-2]
    width, height = frame_list[0].size
    # load 2D spatial
    spa_2D_path = SPATIAL_BASE + f"2D-Pos/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".png"
    spa_2D = Image.open(spa_2D_path).convert("RGB")
    spa_2D = spa_2D.resize((width, height), Image.LANCZOS)
    # load 3D spatial
    spa_3D_path = SPATIAL_BASE + f"3D-Pos/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".png"
    spa_3D = Image.open(spa_3D_path).convert("RGB")
    spa_3D = spa_3D.resize((width, height), Image.LANCZOS)
    frame_list.append(spa_2D)
    frame_list.append(spa_3D)

    # Convert frame_list to numpy array
    video = [np.array(frame) for frame in frame_list]
    video = np.stack(frame_list, axis=0)

    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = num_segments / max(total_frames, 1e-6) * video_fps

    return video, sample_fps, points_info


def _read_video_decord_spa_sti(
    ele: dict,
    points: np.ndarray,
    frame_select_list: list,
    counts: list = None,
    num_segments: int = 8,
) -> (torch.Tensor, float, list):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(frame_select_list), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    # Uniformly sample indices from frame_select_list
    try:
        frame_indices = get_index_with_selection_by_count_diverse(counts, video_fps, total_frames, num_segments=8, bound=None, first_idx=0)
        if len(frame_indices) == 0:
            raise ValueError("Frame indices are empty after count-based sampling.")
    except ValueError as e:
        # Fall back to the original method
        frame_indices = np.linspace(0, len(frame_select_list) - 1, num_segments, dtype=int)
        frame_indices = [frame_select_list[idx] for idx in frame_indices]

    # frame_indices = np.linspace(0, len(frame_select_list) - 1, nframes, dtype=int)
    # frame_indices = [frame_select_list[idx] for idx in frame_indices]

    # Get the corresponding points information for each frame index
    idx_per_frame_indices = [frame_select_list.index(frame_index) for frame_index in frame_indices]
    points_info = [points[frame_select_list.index(frame_index)] for frame_index in frame_indices]

    frame_list = []
    num_points = points.shape[0]
    colors = (plt.cm.viridis(np.linspace(0, 1, num_points))[:, :3] * 255).astype(np.uint8)[:, ::-1]

    for i, frame_index in enumerate(frame_indices):
        frame = vr[frame_index].asnumpy()[:, :, ::-1].copy()  # RGB -> BGR, 直接拷贝为 OpenCV 格式

        color = tuple(int(c) for c in colors[idx_per_frame_indices[i]])  # 取预计算颜色
        center = (frame.shape[1] - 50, 50)
        
        cv2.circle(frame, center, 20, color, -1)

        text = str(frame_index)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # cv2.imwrite("test.png", frame.astype(np.uint8))
        # Draw the points on the image ===

        # Convert back to PIL image
        img = Image.fromarray(frame[:, :, ::-1])
        frame_list.append(img)

    width, height = frame_list[0].size
    # load 2D spatial
    spa_2D_path = SPATIAL_BASE_TSI + f"2D-Pos/all/" + video_path.split("/")[-1].split(".")[0] + ".png"
    spa_2D = Image.open(spa_2D_path).convert("RGB")
    spa_2D = spa_2D.resize((width, height), Image.LANCZOS)
    # load 3D spatial
    spa_3D_path = SPATIAL_BASE_TSI + f"3D-Pos/all/" + video_path.split("/")[-1].split(".")[0] + ".png"
    spa_3D = Image.open(spa_3D_path).convert("RGB")
    spa_3D = spa_3D.resize((width, height), Image.LANCZOS)
    frame_list.append(spa_2D)
    frame_list.append(spa_3D)

    # Convert frame_list to numpy array
    video = [np.array(frame) for frame in frame_list]
    video = np.stack(frame_list, axis=0)

    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = num_segments / max(total_frames, 1e-6) * video_fps

    return video, sample_fps, points_info

VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images

def fetch_video_spa(ele: dict, points: np.ndarray, frame_select_list: list, counts:list, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False, num_segments: int = 8) -> torch.Tensor | list[Image.Image] | list:
    if isinstance(ele["video"], str):
        # Default to decord
        video, sample_fps, points_info = _read_video_decord_spa(ele, points, frame_select_list, counts=counts, num_segments=num_segments)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if return_video_sample_fps:
            return video, sample_fps, points_info
        return video, points_info
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def fetch_video_spa_sti(ele: dict, points: np.ndarray, frame_select_list: list, counts:list, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False, num_segments: int = 8) -> torch.Tensor | list[Image.Image] | list:
    if isinstance(ele["video"], str):
        # Default to decord
        video, sample_fps, points_info = _read_video_decord_spa_sti(ele, points, frame_select_list, counts=counts, num_segments=num_segments)

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if return_video_sample_fps:
            return video, sample_fps, points_info
        return video, points_info
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs

def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs

# only for spatial video
def process_vision_info_spa(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    num_segments: int = 8,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict] | None, list | None]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    # load spatial info
    video_path = vision_infos[0]["video"]
    max_pixels = vision_infos[0]["max_pixels"]
    scene_name = video_path.split("/")[-2]
    spa_2D_path = SPATIAL_BASE + f"2D-Pos/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".png"
    if os.path.exists(spa_2D_path):
        scene_name = video_path.split("/")[-2]
        # load spatial meta info
        spa_path = SPATIAL_BASE + f"points/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".npy"
        spa_info = np.load(spa_path, allow_pickle=True)
        spa_info = {key: spa_info.item().get(key) for key in spa_info.item()}
        
        points = spa_info['points']
        frame_select_list = spa_info['frames_index']
        counts = spa_info['counts']
    
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            if os.path.exists(spa_2D_path):
                video_input, video_sample_fps, points_info = fetch_video_spa(vision_info, points, frame_select_list, 
                                                                             counts=counts, return_video_sample_fps=True, num_segments=num_segments)
            else:
                video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    
    if os.path.exists(spa_2D_path):
        return image_inputs, video_inputs, points_info
    else:
        return image_inputs, video_inputs, None
    

# only for spatial video
def process_vision_info_spa_sti(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    num_segments: int = 8,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict] | None, list | None]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    # load spatial info
    video_path = vision_infos[0]["video"]
    max_pixels = vision_infos[0]["max_pixels"]
    spa_2D_path = SPATIAL_BASE_TSI + f"2D-Pos/all/" + video_path.split("/")[-1].split(".")[0] + ".png"
    if os.path.exists(spa_2D_path):
        scene_name = video_path.split("/")[-2]
        # load spatial meta info
        spa_path = SPATIAL_BASE_TSI + f"points/all/" + video_path.split("/")[-1].split(".")[0] + ".npy"
        spa_info = np.load(spa_path, allow_pickle=True)
        spa_info = {key: spa_info.item().get(key) for key in spa_info.item()}
        
        points = spa_info['points']
        frame_select_list = spa_info['frames_index']
        counts = spa_info['counts']
    
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            if os.path.exists(spa_2D_path):
                video_input, video_sample_fps, points_info = fetch_video_spa_sti(vision_info, points, frame_select_list, 
                                                                             counts=counts, return_video_sample_fps=True, num_segments=num_segments)
            else:
                video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    
    if os.path.exists(spa_2D_path):
        return image_inputs, video_inputs, points_info
    else:
        return image_inputs, video_inputs, None