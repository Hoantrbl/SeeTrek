import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io, color, img_as_ubyte
from ultralytics import YOLO
import time

K = np.array([[718.8560, 0, 607.1928],
              [0, 718.8560, 185.2157],
              [0, 0, 1]])

item_list = ['video', 'points', '3D-Pos', '2D-Pos']
directory = "/home/.cache/huggingface/STI-Bench/video"
mode = "normal"

model_names = ["yolov8n"]
skip_frame_values = [4]

# =========================
# 实用函数
# =========================
def calc_compression_entropy(image_input):
    img = cv2.imread(image_input) if isinstance(image_input, (str, bytes, os.PathLike)) else cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    _, buf = cv2.imencode('.png', img)
    return len(buf) / img.size

def calc_entropy_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    entropies = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        entropy_val = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
        entropies.append(entropy_val)
    return np.mean(entropies)

def calc_local_entropy(image_path, radius=3):
    img = io.imread(image_path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return np.mean(entropy(img_as_ubyte(img), disk(radius)))

def process_video_task(args):
    model_name, skip_frames, key, video_path, save_dir, K, item_list = args
    model = YOLO(model_name + ".pt").to("cpu")

    start_time = time.time()
    try:
        for item in item_list:
            os.makedirs(os.path.join(save_dir, item, key), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"Failed to open {video_path}"

        orb = cv2.ORB_create(2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        trajectory, poses = [np.zeros((3, 1))], [np.eye(3)]
        frame_index, counts = [], []
        frame_num = 0

        ret, prev_frame = cap.read()
        if not ret or prev_frame is None or prev_frame.size == 0:
            cap.release()
            return f"Empty frame in {video_path}"

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % skip_frames != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)
            if des is None or prev_des is None:
                continue

            matches = bf.match(prev_des, des)
            if len(matches) < 8:
                continue

            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, threshold=1.0)
            if E is None:
                continue

            _, R, t, _ = cv2.recoverPose(E, pts2, pts1, K)
            cur_R = poses[-1] @ R
            cur_t = poses[-1] @ t + trajectory[-1]

            trajectory.append(cur_t)
            poses.append(cur_R)
            frame_index.append(frame_num)

            results = model(frame, verbose=False)[0]
            boxes = results.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []
            names = [model.names[cid] for cid in class_ids]
            counts.append([frame_num, names])

            prev_kp, prev_des = kp, des

        trajectory_array = np.array(trajectory).squeeze()
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        np.save(os.path.join(save_dir, 'points', key, base_name + ".npy"), {
            'points': trajectory_array,
            'frames_index': frame_index,
            'counts': counts
        })

        plot_trajectory_3d(trajectory_array, os.path.join(save_dir, '3D-Pos', key, base_name + '.png'))
        plot_trajectory_2d(trajectory_array, os.path.join(save_dir, '2D-Pos', key, base_name + '.png'))
        cap.release()
        elapsed_time = time.time() - start_time
        return f"✅ {video_path}", elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        return f"❌ {video_path} - {str(e)}", elapsed_time

def plot_trajectory_3d(points, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    for i in range(len(points) - 1):
        ax.plot(points[i:i+2, 0], points[i:i+2, 1], points[i:i+2, 2], color=colors[i])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(points)))
    plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10).set_label('Time Step')
    ax.set_title("Estimated 3D Camera Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(path, dpi=300)
    plt.close(fig)

def plot_trajectory_2d(points, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    for i in range(len(points) - 1):
        ax.plot(points[i:i+2, 0], points[i:i+2, 1], color=colors[i])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(points)))
    plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10).set_label('Time Step')
    ax.set_title("Estimated 2D Path (BEV)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.savefig(path, dpi=300)
    plt.close(fig)

folder_dict = {}
folder_dict = {"all": [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mp4")]}
# for root, dirs, files in os.walk(directory):
#     for dir_name in dirs:
#         folder_path = os.path.join(root, dir_name)
#         folder_dict[dir_name] = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for model_name in model_names:
    for skip_frames in skip_frame_values:
        save_dir = f"/home/lipengteng/vsibench/tools/tsibench_{model_name}_2hz_every_{skip_frames}_frames"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        tasks = [(model_name, skip_frames, key, path, save_dir, K, item_list) for key in folder_dict for path in folder_dict[key]]
        if mode == "debug":
            for args in tasks:
                print(f"Processing {args[3]}...")
                print(process_video_task(args))
        else:
            total_time = 0
            video_count = 0

            with ProcessPoolExecutor(max_workers=int(os.cpu_count()*0.5)) as executor:
                futures = [executor.submit(process_video_task, args) for args in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model_name} every {skip_frames} frames"):
                    result, duration = future.result()
                    print(result)
                    total_time += duration
                    video_count += 1

                avg_time = total_time / video_count if video_count > 0 else 0
                print(f"\nAverage processing time per video: {avg_time:.2f} seconds ({video_count} videos)")
