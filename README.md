# SeeTrek
The offcial code base of See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model

🚀 Our paper has been accepted by Neurips 2025.

## Open-Source Plan 
We will open-source our code in the end of November.  

If you want the core but not organized code of our works, direct contact me: 

📧 pli807@connect.hkust-gz.edu.cn

## Abstract
We introduce See\&Trek, the first training-free prompting framework tailored to enhance the spatial understanding of Multimodal Large Language Models MLLMs under vision-only constraints. While prior efforts have incorporated modalities like depth or point clouds to improve spatial reasoning, purely visual-spatial understanding remains underexplored. See\&Trek addresses this gap by focusing on two core principles: increasing visual diversity and motion reconstruction. For visual diversity, we conduct Maximum Semantic Richness Sampling, which employs an off-the-shell perception model to extract semantically rich keyframes that capture scene structure. For motion reconstruction, we simulate visual trajectories and encode relative spatial positions into keyframes to preserve both spatial relations and temporal coherence. Our method is training\&GPU-free, requiring only a single forward pass, and can be seamlessly integrated into existing MLLMs. Extensive experiments on the VSI-Bench and STI-Bench show that See\&Trek consistently boosts various MLLMs performance across diverse spatial reasoning tasks with the most +3.5\% improvement, offering a promising path toward stronger spatial intelligence.

<img width="986" height="306" alt="image" src="https://github.com/user-attachments/assets/7e3cc84c-63fc-4ee0-9158-62daa887c40b" />

## Citation

If you think our work is usefull, please cite our work, I appreciate that. 

```
@article{li2025see,
  title={See\&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model},
  author={Li, Pengteng and Song, Pinhao and Li, Wuyang and Guo, Weiyu and Yao, Huizai and Xu, Yijie and Liu, Dugang and Xiong, Hui},
  journal={arXiv preprint arXiv:2509.16087},
  year={2025}
}
```
