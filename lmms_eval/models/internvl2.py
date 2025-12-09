import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from copy import deepcopy
import os
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        # thumbnail_img = image.resize((image_size, image_size))
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0))
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    valid_indices = [idx for idx in frame_select_list if start_idx <= idx <= end_idx]
    if len(valid_indices) < num_segments:
        raise ValueError("Not enough frames in frame_select_list to satisfy the number of segments.")
    seg_size = float(len(valid_indices)) / num_segments
    frame_indices = np.array([valid_indices[int(np.round(seg_size * idx))] for idx in range(num_segments)])
    return frame_indices


def get_index_with_selection_by_count(
    counts, fps, max_frame, num_segments=32,
    bound=None, first_idx=0
):
    start_time, end_time = bound if bound else (-1e9, 1e9)
    start_idx = max(first_idx, round(start_time * fps))
    end_idx = min(round(end_time * fps), max_frame)

    valid = [(f, c) for f, c in counts if start_idx <= f <= end_idx]
    if len(valid) < num_segments:
        raise ValueError("Not enough valid frames in time bound.")

    valid.sort(key=lambda x: x[0])
    frame_nums = [f for f, _ in valid]
    object_counts = [c for _, c in valid]

    total = len(valid)
    seg_len = total // num_segments
    selected = []

    for i in range(num_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < num_segments - 1 else total
        segment = list(range(start, end))
        if not segment:
            continue
        # Select the frame with the maximum object count in the segment, preferring later frames in case of a tie
        best_idx = max(segment, key=lambda idx: (object_counts[idx], -frame_nums[idx]))
        selected.append(frame_nums[best_idx])

    return np.array(selected)

def get_topk_indices_by_count(
    counts, fps, max_frame, num_segments=8,
    bound=None, first_idx=0
):
    start_time, end_time = bound if bound else (-1e9, 1e9)
    start_idx = max(first_idx, round(start_time * fps))
    end_idx = min(round(end_time * fps), max_frame)

    valid = [(f, names) for f, names in counts if start_idx <= f <= end_idx]
    if len(valid) < num_segments:
        raise ValueError("Not enough valid frames to select from.")

    # Sort by object count (descending), then by frame number (ascending)
    valid.sort(key=lambda x: (-len(x[1]), x[0]))
    # Select the top K frames
    topk_frames = sorted(f for f, _ in valid[:num_segments])

    return np.array(topk_frames)


def get_index_with_selection_by_count_diverse(
    counts, fps, max_frame, num_segments=8,
    bound=None, first_idx=0
):
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

    # Select the global best frame (max count, min frame num)
    global_best_idx = max(range(len(valid)), key=lambda i: (object_counts[i], -frame_nums[i]))
    selected_indices = [global_best_idx]
    selected_classes = class_names[global_best_idx]

    total = len(valid)
    remaining_segments = num_segments - 1
    segment_len = total // remaining_segments if remaining_segments > 0 else total

    for i in range(remaining_segments):
        start = i * segment_len
        end = (i + 1) * segment_len if i < remaining_segments - 1 else total
        segment = list(range(start, end))
        if not segment:
            continue

        # Select the best frame in the segment for diversity (min intersection with selected classes)
        # Tie-breaking: max object count, min frame num
        best_idx = min(
            segment,
            key=lambda idx: (
                len(class_names[idx] & selected_classes),  # Minimize class intersection
                -object_counts[idx],                      # Maximize object count
                frame_nums[idx]                           # Minimize frame number
            )
        )
        selected_indices.append(best_idx)
        selected_classes.update(class_names[best_idx])

    return np.array(sorted([frame_nums[i] for i in selected_indices]))


from datetime import timedelta

from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs


@register_model("internvl2")
class InternVL2(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL2-2B",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        max_frames_num: int = 32,
        min_gap: int = 48,
        use_spatial: bool = False,
        spatial_choice: list = ['2D','3D','points'],
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained

        # make sure internvl2 works well with pipeline parallel
        # self._model.generate = types.MethodType(generate, self._model)
        if device_map == 'auto':
            self._model = AutoModel.from_pretrained(self.path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map=device_map).eval()
            from accelerate.hooks import add_hook_to_module, AlignDevicesHook
            add_hook_to_module(self._model.language_model.lm_head, AlignDevicesHook(execution_device=self._model.language_model.model.embed_tokens.weight.device))
        else:
            self._model = AutoModel.from_pretrained(self.path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()

        self._tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        # self._model.tokenizer = self._tokenizer

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for InternVL2, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.modality = modality
        self.max_frames_num = max_frames_num
        self.min_gap = min_gap

        # aims for utilizing spatial info.
        self.use_spatial = use_spatial
        self.spatial_choice = spatial_choice
        self.spatial_base = "./tools/SeeTrek/modified_dataset_yolov8n_2hz_every_4_frames/"
        # prompt setting
        self.spatial_prompt_universal = (
            "Each video frame has its serial number in the top-right corner. "
            "The frame’s highlight color matches the color in the spatial map, indicating its position."
        )
        self.spatial_prompt_2D_3D = (
            "Both 2D (bird's-eye) and 3D views illustrate the camera's spatial trajectory, "
            "with color encoding time progression."
        )
        self.spatial_prompt_points = (
            "Points represent the camera's relative positions; the number of points reflects only spatial relationships."
        )

        if self.use_spatial:
            self.prompt_spatial = ""
            self.prompt_spatial += self.spatial_prompt_universal + "\n\n" + self.spatial_prompt_2D_3D + "\n\n" + self.spatial_prompt_points
    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video_spatial(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0))
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)

        scene_name = video_path.split("/")[-2]
        # load spatial meta info
        spa_path = self.spatial_base + f"points/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".npy"
        spa_info = np.load(spa_path, allow_pickle=True)
        spa_info = {key: spa_info.item().get(key) for key in spa_info.item()}
        points = spa_info['points']
        frame_select_list = spa_info['frames_index']
        counts = spa_info['counts']
        # entropies = spa_info['entropy']
        # load 2D spatial
        spa_2D_path = self.spatial_base + f"2D-Pos/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".png"
        # load 3D spatial
        spa_3D_path = self.spatial_base + f"3D-Pos/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".png"

        # frame_indices = get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=num_segments)

        try:
            frame_indices = get_index_with_selection_by_count_diverse(counts, fps, max_frame, num_segments=num_segments, bound=bound, first_idx=0)
            # frame_indices = get_topk_indices_by_count(counts, fps, max_frame, num_segments=num_segments, bound=bound, first_idx=0)
            if len(frame_indices) == 0:
                raise ValueError("Frame indices are empty after count-based sampling.")
        except Exception as e:
            eval_logger.warning(f"Count-based sampling failed with error: {e}. Falling back to get_index_with_selection.")
            # Fall back to the original method
            frame_indices = get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=num_segments)
        # frame_indices = get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=num_segments)

        # Get the corresponding points information for each frame index
        idx_per_frame_indices = [frame_select_list.index(frame_index) for frame_index in frame_indices]
        points_info = [points[frame_select_list.index(frame_index)] for frame_index in frame_indices]

        for i, frame_index in enumerate(frame_indices):
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            # Draw the points on the image ===
            num_points = points.shape[0]
            colors = plt.cm.viridis(np.linspace(0, 1, num_points))
            color = tuple((colors[idx_per_frame_indices[i]][:3] * 255).astype(int)[::-1])  # Convert to BGR and ensure integer type
            # Convert PIL image to OpenCV format
            frame = deepcopy(np.array(img)[:, :, ::-1])
            center = (frame.shape[1] - 50, 50)  # Top-right corner
            radius = 20
            thickness = -1  # Filled circle
            # Draw circle
            cv2.circle(frame, center, radius, tuple(map(int, color)), thickness)
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (255, 255, 255)  # White
            text_thickness = 2
            text = str(frame_index)
            text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, tuple(map(int, text_color)), text_thickness)
            # cv2.imwrite("test.png", frame.astype(np.uint8))
            # Draw the points on the image ===

            # Convert back to PIL image
            img = Image.fromarray(frame[:, :, ::-1])
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        # Add the spatial images to the pixel values
        if "3D" in self.spatial_choice or "all" in self.spatial_choice:
            spa_3D = Image.open(spa_3D_path).convert("RGB")
            spa_3D = [transform(spa_3D)]
            spa_3D = torch.stack(spa_3D)
            num_patches_list.append(spa_3D.shape[0])
            pixel_values_list.append(spa_3D)

        if "2D" in self.spatial_choice or "all" in self.spatial_choice:
            spa_2D = Image.open(spa_2D_path).convert("RGB")
            spa_2D = [transform(spa_2D)]
            spa_2D = torch.stack(spa_2D)
            num_patches_list.append(spa_2D.shape[0])
            pixel_values_list.append(spa_2D)

        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list, points_info
    def load_video_spatial_V2(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0))
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)

        scene_name = video_path.split("/")[-2]
        # load spatial meta info
        spa_path = self.spatial_base + f"points/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".npy"
        spa_info = np.load(spa_path, allow_pickle=True)
        spa_info = {key: spa_info.item().get(key) for key in spa_info.item()}
        points = spa_info['points']
        frame_select_list = spa_info['frames_index']
        counts = spa_info['counts']
        # entropies = spa_info['entropy']
        # load 2D spatial
        spa_2D_path = self.spatial_base + f"2D-Pos/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".png"
        # load 3D spatial
        spa_3D_path = self.spatial_base + f"3D-Pos/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".png"

        # frame_indices = get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=num_segments)
        try:
            frame_indices = get_index_with_selection_by_count_diverse(counts, fps, max_frame, num_segments=num_segments, bound=bound, first_idx=0)
            # frame_indices = get_topk_indices_by_count(counts, fps, max_frame, num_segments=num_segments, bound=bound, first_idx=0)
            if len(frame_indices) == 0:
                raise ValueError("Frame indices are empty after count-based sampling.")
        except Exception as e:
            eval_logger.warning(f"Count-based sampling failed with error: {e}. Falling back to get_index_with_selection.")
            # Fall back to the original method
            frame_indices = get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=num_segments)
        # frame_indices = get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=num_segments)

        # Get the corresponding points information for each frame index
        idx_per_frame_indices = [frame_select_list.index(frame_index) for frame_index in frame_indices]
        points_info = [points[frame_select_list.index(frame_index)] for frame_index in frame_indices]

        num_points = points.shape[0]
        colors = (plt.cm.viridis(np.linspace(0, 1, num_points))[:, :3] * 255).astype(np.uint8)[:, ::-1]

        for i, frame_index in enumerate(frame_indices):
            frame = vr[frame_index].asnumpy()[:, :, ::-1].copy()

            color = tuple(int(c) for c in colors[idx_per_frame_indices[i]])
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
            pixel_values = [transform(tile) for tile in [img]]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)

        # Add the spatial images to the pixel values
        if "3D" in self.spatial_choice or "all" in self.spatial_choice:
            spa_3D = Image.open(spa_3D_path).convert("RGB")
            spa_3D = [transform(spa_3D)]
            spa_3D = torch.stack(spa_3D)
            num_patches_list.append(spa_3D.shape[0])
            pixel_values_list.append(spa_3D)

        if "2D" in self.spatial_choice or "all" in self.spatial_choice:
            spa_2D = Image.open(spa_2D_path).convert("RGB")
            spa_2D = [transform(spa_2D)]
            spa_2D = torch.stack(spa_2D)
            num_patches_list.append(spa_2D.shape[0])
            pixel_values_list.append(spa_2D)

        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list, points_info

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if visuals != [None]:
                visuals = self.flatten(visuals)
                if self.modality == "image":
                    if visuals:
                        visuals = [load_image(visual).to(torch.bfloat16).cuda() for visual in visuals]
                        pixel_values = torch.cat(visuals, dim=0)
                        num_patches_list = [visual.size(0) for visual in visuals]
                        image_tokens = ["<image>"] * len(visuals)
                        image_tokens = " ".join(image_tokens)
                        contexts = image_tokens + "\n" + contexts
                    else:
                        pixel_values = None
                        num_patch_list = None
                    response, history = self.model.chat(self.tokenizer, pixel_values, contexts, gen_kwargs, num_patches_list=num_patches_list, history=None, return_history=True)
                elif self.modality == "video":
                    assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                    video_path = visuals[0]
                    scene_name = video_path.split("/")[-2]
                    spa_2D_path = self.spatial_base + f"2D-Pos/{scene_name}/" + video_path.split("/")[-1].split(".")[0] + ".png"

                    if self.use_spatial and os.path.exists(spa_2D_path):
                        pixel_values, num_patches_list, points_info = self.load_video_spatial_V2(video_path, num_segments=self.max_frames_num, max_num=1)
                        pixel_values = pixel_values.to(torch.bfloat16).cuda()
                        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

                        if "points" in self.spatial_choice or "all" in self.spatial_choice:
                            points_info = [f"Frame{i+1}: {points_info[i]}" for i in range(len(points_info))]
                            points_info = " ".join(points_info)
                            # question = self.prompt_spatial + points_info + "\n\n" + self.prompt_bridge + "\n\n" + video_prefix + contexts
                            question = self.prompt_spatial + points_info + "\n\n" + video_prefix + contexts
                            # print(question)
                        else:
                            question = self.prompt_spatial + video_prefix + contexts
                    else:
                        pixel_values, num_patches_list = load_video(video_path, num_segments=self.max_frames_num, max_num=1)
                        pixel_values = pixel_values.to(torch.bfloat16).cuda()
                        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                        question = video_prefix + contexts
                    response, history = self.model.chat(self.tokenizer, pixel_values, question, gen_kwargs, num_patches_list=num_patches_list, history=None, return_history=True)
            else:
                response, history = self.model.chat(self.tokenizer, None, contexts, gen_kwargs, num_patches_list=None, history=None, return_history=True)
            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."