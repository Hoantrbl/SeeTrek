# SeeTrek
The offcial code base of *See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model*

🚀 Our paper has been accepted by Neurips 2025.


# Install

Please directly follow the [VSI-Benchmark](https://github.com/vision-x-nyu/thinking-in-space) official instruction.

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

Note that we utilize `transformers==4.52.0, lmms_eval==0.2.3, torch==2.6.0, torchvision==0.21.0`


Then you need to install the [Ultralytics Yolo](https://github.com/ultralytics/ultralytics):

```cmd
pip install ultralytics
```

# Run

1. Before running our method, you need to *make sure that the VSI-Benchmark dataset has been downloaded locally.* Next, make sure the path setting in `tools/multi-proc-video-skip.py`, like `directory` and `save_dir`. 
2. Download the checkpoint like Internvl3-8B in Folder `checkpoint-local` and modify the your ckpt path in `model_args` from `./evaluate.sh`
3. Make sure the saved prior semantcis dir has been set turely in its model file like `self.spatial_base` in `lmms_eval/models/internvl2.py` 

First you need to generate the prior semantics about given video:

```cmd
python ./tools/multi-proc-video-skip.py
```
all prior semantics would be processed and saved on the local dir. 

Then, you can run the following command to get the final results. `num_processes` denotes that utilize 2 gpus to inference, you can modify it according to your gpus:

```cmd
bash evaluate.sh --model all --num_processes 2
```


# Citation

If you think our work is usefull, please cite our work, I appreciate that. 

```
@article{li2025see,
  title={See\&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model},
  author={Li, Pengteng and Song, Pinhao and Li, Wuyang and Guo, Weiyu and Yao, Huizai and Xu, Yijie and Liu, Dugang and Xiong, Hui},
  journal={arXiv preprint arXiv:2509.16087},
  year={2025}
}
```
