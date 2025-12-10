# SeeTrek

> **Training-Free Spatial Prompting for Multimodal Large Language Models**

<p align="left">
  <img src="https://img.shields.io/badge/NeurIPS-2025-8A2BE2.svg"/>
  <img src="https://img.shields.io/badge/Status-Available-brightgreen"/>
  <img src="https://img.shields.io/badge/License-MIT-blue"/>
</p>

Official codebase for <u>*See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Models*</u>

🎉 Our paper has been accepted to **NeurIPS 2025**.

---

## 📝 Abstract

**See&Trek** is a **training-free** and **GPU-free** spatial prompting framework designed to fundamentally enhance the spatial reasoning capabilities of Multimodal Large Language Models (MLLMs). It addresses two key bottlenecks in existing MLLMs:
**(1) visual homogeneity** caused by uniform frame sampling, and
**(2) unknown motion** due to missing ego-motion cues.

See&Trek achieves this by (i) extracting **semantic-rich keyframes** using off-the-shelf perception models and (ii) reconstructing **camera trajectories** via Visual Odometry to annotate keyframes with explicit motion information. Without any model modification or fine-tuning, See&Trek injects structured spatial–temporal priors into MLLMs through a **single forward pass**, leading to robust improvements across spatial reasoning tasks.

---

## 🌟 Highlights

* 🚫 **Training-free & GPU-free** spatial prompting
* 🔌 **Plug-and-play** for all open-source and commercial MLLMs
* ⚡ **Single-forward** inference with zero architecture changes
* 🎥 **Semantic-rich keyframe selection + reconstructed motion cues**
* 📈 Consistent gains on **VSI-Bench** and **STI-Bench**

---

# 📦 Installation

Follow the official setup for **VSI-Benchmark**:
[https://github.com/vision-x-nyu/thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space)

```cmd
conda create --name vsibench python=3.10
conda activate vsibench

git clone git@github.com:vision-x-nyu/thinking-in-space.git
cd thinking-in-space

git submodule update --init --recursive

cd transformers && pip install -e . && cd ..

pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
pip install deepspeed
```

**Core dependencies:**

```
transformers==4.52.0
lmms_eval==0.2.3
torch==2.6.0
torchvision==0.21.0
```

Install **Ultralytics YOLO**:

```cmd
pip install ultralytics
```

---

# 🚀 Running SeeTrek

## 1️⃣ Generate Prior Semantics

Make sure dataset paths (e.g., `directory`, `save_dir`) in
`tools/multi-proc-video-skip.py` are correctly configured.

Run:

```cmd
python ./tools/multi-proc-video-skip.py
```

Outputs are saved to:

```
./tools/SeeTrek/modified_dataset_yolov8n_2hz_every_4_frames
```

We also provide **preprocessed prior semantics** on Google Drive:
🔗 [https://drive.google.com/drive/folders/1X3ZpaO8H59H-OxoEturHi2AwWHpm8FMu?usp=drive_link](https://drive.google.com/drive/folders/1X3ZpaO8H59H-OxoEturHi2AwWHpm8FMu?usp=drive_link)

---

## 2️⃣ Run Evaluation

Place model checkpoints (e.g., InternVL3-8B) into `checkpoint-local`
and update paths in `evaluate.sh`.

Then run:

```cmd
bash evaluate.sh --model all --num_processes 2
```

`--num_processes` controls how many GPUs to use.

---

# 📚 Citation

If you find See&Trek useful, please cite:

```
@article{li2025see,
  title={See\&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model},
  author={Li, Pengteng and Song, Pinhao and Li, Wuyang and Guo, Weiyu and Yao, Huizai and Xu, Yijie and Liu, Dugang and Xiong, Hui},
  journal={arXiv preprint arXiv:2509.16087},
  year={2025}
}
```
