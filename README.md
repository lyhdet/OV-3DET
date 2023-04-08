# OV-3DET: Open-Vocabulary Point-Cloud Object Detection without 3D Annotation

**OV-3DET**: A **O**pen **V**ocabulary **3**D **DET**ector. 

[Paper](https://arxiv.org/abs/2304.00788) | [BibTeX](#citation)

 <p align="center"> <img src='Assets/overview.png' align="center" height="300px"> </p>

>[**OV-3DET: Open-Vocabulary Point-Cloud Object Detection without 3D Annotation**](https://arxiv.org/abs/2304.00788),                                                
>Yuheng Lu, Chenfeng Xu, Xiaobao Wei, Xiaodong Xie, Masayoshi Tomizuka, Kurt Keutzer and Shanghang Zhang,                                                               
>Accepted to *CVPR2023*                                                 
 
 ## Features
- Detects 3D objects according to text prompting.

- The training of OV-3DET does not require 3D annotation.


## Installation
See [installation instructions](Prepare_ScanNet.md).


##  Training OV-3DET
### Phase 1 
&emsp;Generating 3D pseudo box for training localization. You can generate pseudo-label of ScanNet by:

1. Prepare the [ScanNet](https://github.com/lyhdet/OV-3DET/blob/main/Prepare_ScanNet.md) dataset.
2. Moving:  "***domo.py***, ***scannet_pseudo_make.sh***, ***scannet_util.py*** " to the ***[Detic](https://github.com/facebookresearch/Detic)*** codebase.
3. Run the ***scannet_pseudo_make.sh***:    `bash scannet_pseudo_make.sh`
4. Replace the ground truth box of the training set with pseudo label.

### Phase 2 
&emsp;Connecting the embedding space of ***Text, Image and Point-cloud*** by running:  `bash scripts/scannet_quick_lr_7e-4.sh`


##  Test OV-3DET
To evaluate OV-3DET, simply by running: `bash scripts/evaluate.sh`

## Acknowledgement
This codebase is modified base on ***3DETR*** [1], ***CLIP*** [2] and ***Detic*** [3], we sincerely appreciate their contributions!

>[1] An end-to-end transformer model for 3d object detection. *ICCV*. 2021.                                                                                             
>[2] Learning transferable visual models from natural language supervision. *ICML*. 2021.                                                              
>[3] Detecting twenty-thousand classes using image-level supervision. *ECCV*. 2022.                                                                                             

## Citation
If you find this repository helpful, please consider citing our work:

```
@article{lu2023open,
  title={Open-Vocabulary Point-Cloud Object Detection without 3D Annotation},
  author={Lu, Yuheng and Xu, Chenfeng and Wei, Xiaobao and Xie, Xiaodong and Tomizuka, Masayoshi and Keutzer, Kurt and Zhang, Shanghang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
