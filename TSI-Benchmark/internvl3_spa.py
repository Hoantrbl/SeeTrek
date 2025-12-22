#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------- #
#                               Imports & Setup                               #
# --------------------------------------------------------------------------- #
import os, re, json, signal, warnings
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import AutoConfig
import math
from copy import deepcopy

# Specify GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"


# --------------------------------------------------------------------------- #
#                               Configuration                                 #
# --------------------------------------------------------------------------- #
MODEL_ID     = "InternVL3-8B"
MODEL_PATH   = f"./checkpoint-local/{MODEL_ID}"
PARQUET_FILE = "/home/.cache/huggingface/STI-Bench/qa.parquet"
VIDEO_DIR    = "/home/.cache/huggingface/STI-Bench/video"
RESULTS_JSON = f"./TSI-Benchmark/output/{MODEL_ID}_spa.json"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FRAMES   = 8   # Target frames for time alignment

spatial_base = "./tools/tsibench_yolov8n_2hz_every_4_frames/"
spatial_choice = ['all']

# Spatial prompt instructions for the model
spatial_prompt_universal = (
    "Each video frame has its serial number in the top-right corner. "
    "The frame’s highlight color matches the color in the spatial map, indicating its position."
)
spatial_prompt_2D_3D = (
    "Both 2D (bird's-eye) and 3D views illustrate the camera's spatial trajectory, "
    "with color encoding time progression."
    "Please ignores the value of the axis, which only represents the camera's relative position."
)
spatial_prompt_points = (
    "Points represent the camera's relative positions; the number of points reflects only spatial relationships."
)
prompt_spatial = spatial_prompt_universal + spatial_prompt_2D_3D + spatial_prompt_points


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")                      # Set plot style

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the target aspect ratios allowed
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the original image
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and crop into tiles
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    """Logic to distribute model layers across multiple GPUs."""
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU handles the Vision Transformer (ViT), treat it as half a GPU for LLM layers.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Uniformly samples frame indices."""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=32):
    """Samples frames from a pre-selected list within the time bounds."""
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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def load_video_spatial(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Loads video frames augmented with spatial data indicators."""
    vr = VideoReader(video_path, ctx=cpu(0))
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    # Load spatial meta info (trajectories/points)
    spa_path = spatial_base + f"points/all/" + video_path.split("/")[-1].split(".")[0] + ".npy"
    spa_info = np.load(spa_path, allow_pickle=True)
    spa_info = {key: spa_info.item().get(key) for key in spa_info.item()}
    points = spa_info['points']
    frame_select_list = spa_info['frames_index']
    counts = spa_info['counts']
    
    # Load 2D/3D spatial map images
    spa_2D_path = spatial_base + f"2D-Pos/all/" + video_path.split("/")[-1].split(".")[0] + ".png"
    spa_3D_path = spatial_base + f"3D-Pos/all/" + video_path.split("/")[-1].split(".")[0] + ".png"

    # Select frame indices using diversity-based sampling
    try:
        frame_indices = get_index_with_selection_by_count_diverse(counts, fps, max_frame, num_segments=num_segments, bound=bound, first_idx=0)
        if len(frame_indices) == 0:
            raise ValueError("Frame indices are empty after count-based sampling.")
    except Exception as e:
        # Fall back to the standard selection method
        frame_indices = get_index_with_selection(bound, fps, max_frame, frame_select_list, first_idx=0, num_segments=num_segments)

    # Get points info for chosen frames
    idx_per_frame_indices = [frame_select_list.index(frame_index) for frame_index in frame_indices]
    points_info = [points[frame_select_list.index(frame_index)] for frame_index in frame_indices]

    for i, frame_index in enumerate(frame_indices):
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        
        # Draw visual indicators (color-coded circle and frame index) on the frame
        num_points = points.shape[0]
        colors = plt.cm.viridis(np.linspace(0, 1, num_points))
        color = tuple((colors[idx_per_frame_indices[i]][:3] * 255).astype(int)[::-1])  # BGR for OpenCV
        
        frame = deepcopy(np.array(img)[:, :, ::-1])
        center = (frame.shape[1] - 50, 50)  # Position: Top-right corner
        radius = 20
        thickness = -1  # Filled circle
        cv2.circle(frame, center, radius, tuple(map(int, color)), thickness)

        # Overlay frame number text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (255, 255, 255)  # White
        text_thickness = 2
        text = str(frame_index)
        
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, tuple(map(int, text_color)), text_thickness)

        # Process the augmented image back to PIL and tensor
        img = Image.fromarray(frame[:, :, ::-1])
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    # Append the spatial map images (3D/2D views) to the pixel value list
    if "3D" in spatial_choice or "all" in spatial_choice:
        spa_3D = Image.open(spa_3D_path).convert("RGB")
        spa_3D = [transform(spa_3D)]
        spa_3D = torch.stack(spa_3D)
        num_patches_list.append(spa_3D.shape[0])
        pixel_values_list.append(spa_3D)

    if "2D" in spatial_choice or "all" in spatial_choice:
        spa_2D = Image.open(spa_2D_path).convert("RGB")
        spa_2D = [transform(spa_2D)]
        spa_2D = torch.stack(spa_2D)
        num_patches_list.append(spa_2D.shape[0])
        pixel_values_list.append(spa_2D)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, points_info


def get_index_with_selection_by_count_diverse(
    counts, fps, max_frame, num_segments=8,
    bound=None, first_idx=0
):
    """
    Selects keyframes based on object diversity: prioritizing the frame with the global maximum 
    number of objects, and then selecting frames across segments with the least class overlap.
    """
    # Step 1: Filter by time range
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

    # Step 2: Select the global frame with the most objects (tie-breaker: earlier frame)
    global_best_idx = max(range(len(valid)), key=lambda i: (object_counts[i], -frame_nums[i]))
    selected_indices = [global_best_idx]
    selected_classes = class_names[global_best_idx]

    # Step 3: Select remaining frames across segments to maximize class diversity
    total = len(valid)
    remaining_segments = num_segments - 1
    segment_len = total // remaining_segments if remaining_segments > 0 else total

    for i in range(remaining_segments):
        start = i * segment_len
        end = (i + 1) * segment_len if i < remaining_segments - 1 else total
        segment = list(range(start, end))
        if not segment:
            continue

        # Criteria: Minimum overlap with already selected classes, then max object count
        best_idx = min(
            segment,
            key=lambda idx: (
                len(class_names[idx] & selected_classes),         # Least overlap
                -object_counts[idx],                              # Most objects
                frame_nums[idx]                                   # Earlier frame
            )
        )
        selected_indices.append(best_idx)
        selected_classes.update(class_names[best_idx])

    # Return sorted frame numbers
    return np.array(sorted([frame_nums[i] for i in selected_indices]))

# --------------------------------------------------------------------------- #
#                            Timeout (per sample)                             #
# --------------------------------------------------------------------------- #
class TimeoutException(Exception): ...

def _timeout_handler(signum, frame): raise TimeoutException()
signal.signal(signal.SIGALRM, _timeout_handler)

# --------------------------------------------------------------------------- #
#                   Compute sample_fps without re-encoding                    #
# --------------------------------------------------------------------------- #
def compute_sample_fps(video_path: str, max_frames: int = MAX_FRAMES) -> float:
    """Return FPS that yields ≤ max_frames evenly-spaced frames over the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / orig_fps if orig_fps else total / 1.0
    cap.release()

    if total <= max_frames or duration == 0:
        return orig_fps
    return max_frames / duration

# --------------------------------------------------------------------------- #
#                      Robust answer extraction utility                       #
# --------------------------------------------------------------------------- #
ANSWER_PATTERNS = [
    r"\(([A-E])\)",                                # (A)
    r"Ans\s*=\s*['\"]?([A-E])['\"]?",              # Ans='C'
    r"Answer\s*[:=]\s*([A-E])",                    # Answer: B
    r"Option\s+([A-E])",                           # Option D
    r"\b([A-E])\s*(?:is|was)\s*correct",           # A is correct
    r"\b([A-E])[\.\)]\s*$",                        # C.  /  D)
    r"\b([A-E])!",                                 # A!
]

def extract_answer(text: str) -> str | None:
    for pat in ANSWER_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m: return m.group(1).upper()
    return None

# --------------------------------------------------------------------------- #
#                           Scene categorization                              #
# --------------------------------------------------------------------------- #
def scene_type(filename: str) -> str:
    """Classifies scene based on filename patterns."""
    if "camera" in filename:
        return "outdoor"
    if filename.startswith("scene"):
        return "indoor"
    if filename.split(".")[0].isdigit() and len(filename.split(".")[0]) == 6:
        return "desktop"
    return "unknown"

# --------------------------------------------------------------------------- #
#                        Load model & processor                               #
# --------------------------------------------------------------------------- #
print(f"[Init] Loading model: {MODEL_ID}")
device_map = split_model('InternVL3-8B')
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=128, do_sample=True)
print("[Init] Model & processor ready.\n")

# --------------------------------------------------------------------------- #
#                          Read dataset from parquet                          #
# --------------------------------------------------------------------------- #
print(f"[Data] Reading {PARQUET_FILE}")
df_parquet = pd.read_parquet(PARQUET_FILE)
items = df_parquet.to_dict(orient="records")

for it in items:
    it["file"] = it.get("Video", "unknown.mp4")
    if isinstance(it.get("Candidates"), str):
        try: it["Candidates"] = json.loads(it["Candidates"])
        except json.JSONDecodeError: it["Candidates"] = {}
print(f"[Data] {len(items)} records loaded.\n")

# --------------------------------------------------------------------------- #
#                    Resume previous results if they exist                    #
# --------------------------------------------------------------------------- #
if os.path.exists(RESULTS_JSON):
    try:
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[Resume] Loaded {len(results)} previous results.\n")
    except Exception:
        results = []
else: results = []

# --------------------------------------------------------------------------- #
#                               Inference loop                                #
# --------------------------------------------------------------------------- #
print("[Run] Starting inference …")
for idx, entry in enumerate(items, 1):
    vid_name, sample_id = entry["file"], entry.get("ID")

    # Check if sample already processed
    if any(r["id"] == vid_name and r["id_file"] == sample_id for r in results):
        print(f"[Skip] {idx}/{len(items)}  ({vid_name}, ID={sample_id})")
        continue

    print(f"[Run ] {idx}/{len(items)}  ({vid_name}, ID={sample_id})")
    video_path = os.path.join(VIDEO_DIR, vid_name)
    if not os.path.exists(video_path):
        print(f"       ! video not found → {video_path}")
        continue

    # Prepare prompt text
    cand_str = "\n".join([f"({k}) {v}" for k, v in entry["Candidates"].items()])
    ts, te = entry["time_start"], entry["time_end"]
    prompt_txt = (
        f"From {ts} s to {te} s. {entry['Question']}\n"
        f"{cand_str}\nPlease output only the option letter!"
    )

    # Process video with spatial info
    try: 
        pixel_values, num_patches_list, points_info = load_video_spatial(video_path, num_segments=MAX_FRAMES, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    except Exception as e:
        print(f"       ! video processing error: {e}")
        continue

    signal.alarm(60)  # Set 60 second timeout per sample
    try:
        points_info_str = [f"Frame{i+1}: {points_info[i]}" for i in range(len(points_info))]
        points_info_str = " ".join(points_info_str)
        # Combine spatial instructions, point data, and video frames into final query
        question = prompt_spatial + points_info_str + "\n\n" + video_prefix + prompt_txt
        
        raw_response = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=False)
        response = extract_answer(raw_response) or raw_response.strip()
        print(f'>    Raw: {raw_response}')
        print(f'>    Response: {response}')

        results.append({
            "id": vid_name,
            "id_file": sample_id,
            "time_s": ts,
            "time_e": te,
            "scene": entry.get("scene") or entry.get("Scene"),
            "model_out": response,
            "ans": entry.get("Answer"),
            "Task": entry.get("Task"),
            "que": prompt_txt,
        })

    except TimeoutException:
        print("       ! timeout (>60 s)")
    except Exception as e:
        print(f"       ! error: {e}")
    finally:
        signal.alarm(0)
        torch.cuda.empty_cache()

    # Periodic save
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"[Save] {len(results)} results → {RESULTS_JSON}\n")

print(f"[Done] Inference completed — total saved: {len(results)}\n")

# --------------------------------------------------------------------------- #
#                                ANALYSIS                                     #
# --------------------------------------------------------------------------- #
print("[Ana ] Generating statistics & plots …")
out_dir = f"analysis_results_{MODEL_ID.replace('/','_')}"
os.makedirs(out_dir, exist_ok=True)

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))

# Ensure scene info exists
if "scene" not in df.columns:
    df["scene"] = df["id"].apply(scene_type)
else:
    df["scene"] = df["scene"].fillna(df["id"].apply(scene_type))

# Compute correctness
df["correct"] = df.apply(
    lambda r: (r["model_out"][-1] in "ABCDE") and (r["model_out"][-1] == r["ans"]),
    axis=1,
)

# --- Radar Charts (Overall + Per Scene) ---
tasks = sorted(df["Task"].unique())
task_acc_overall = df.groupby("Task")["correct"].mean().reindex(tasks, fill_value=0)
task_cnt_overall = df.groupby("Task")["correct"].count().reindex(tasks, fill_value=0)

def plot_radar(ax, labels, values, title, counts):
    filtered = [(l, v, c) for l, v, c in zip(labels, values, counts) if c > 0]
    if not filtered:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no samples)")
        return

    labels, values, _ = zip(*filtered)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    vals   = list(values) + [values[0]]
    angs   = list(angles) + [angles[0]]

    ax.plot(angs, vals, marker="o")
    ax.fill(angs, vals, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([.2,.4,.6,.8,1.0])
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.1)

scenes = ["indoor", "desktop", "outdoor"]
fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True), figsize=(14, 12))
plot_radar(axes[0, 0], tasks, task_acc_overall, "Overall", task_cnt_overall)
for ax, scn in zip(axes.flat[1:], scenes):
    sub = df[df["scene"] == scn]
    acc = sub.groupby("Task")["correct"].mean().reindex(tasks, fill_value=0)
    cnt = sub.groupby("Task")["correct"].count().reindex(tasks, fill_value=0)
    plot_radar(ax, tasks, acc, scn.capitalize(), cnt)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "radar_charts.png"), dpi=300)
plt.close()

# --- Bar Plot: Correct vs Total for each scene ---
correct_cnt = df.groupby("scene")["correct"].sum().reindex(scenes, fill_value=0)
total_cnt   = df.groupby("scene")["correct"].count().reindex(scenes, fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
idx = np.arange(len(scenes))
bar_w = 0.35
ax.bar(idx, correct_cnt, bar_w, label="Correct")
ax.bar(idx + bar_w, total_cnt, bar_w, label="Total")

for i, (c, t) in enumerate(zip(correct_cnt, total_cnt)):
    acc = c / t if t else 0
    ax.text(i + bar_w/2, t + 1, f"{acc:.1%}", ha="center")

ax.set_xticks(idx + bar_w/2)
ax.set_xticklabels([s.capitalize() for s in scenes])
ax.set_title("Correct / Total by Scene")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "scene_counts.png"), dpi=300)
plt.close()

# --- Final Textual Report ---
overall_acc = df["correct"].mean()
task_acc  = df.groupby("Task")["correct"].mean().sort_values(ascending=False)
scene_acc = df.groupby("scene")["correct"].mean().sort_values(ascending=False)

report = [
    "=== Performance Report ===",
    f"Total samples: {len(df)}",
    f"Overall accuracy: {overall_acc:.2%}\n",
    "Accuracy by task:",
] + [f"  {t}: {a:.2%}" for t, a in task_acc.items()] + [
    "\nAccuracy by scene:",
] + [f"  {s}: {a:.2%}" for s, a in scene_acc.items()]

report_txt = "\n".join(report)
print(report_txt)

with open(os.path.join(out_dir, "analysis_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_txt)

print(f"[Ana ] Artifacts saved under: {out_dir}")
print("[Done] All stages complete.")