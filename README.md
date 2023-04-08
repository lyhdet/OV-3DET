# OV-3DET: Open-Vocabulary Point-Cloud Object Detection without 3D Annotation

Accepted to CVPR2023. 

[Paper](https://arxiv.org/abs/2304.00788) | [BibTeX](#citation)
 
 &emsp;
 
<img src="Assets/overview.png" width="100%">


 &emsp;The goal of open-vocabulary detection is to identify novel objects based on arbitrary textual descriptions. In this paper, we address open-vocabulary 3D point-cloud detection by a dividing-and-conquering strategy, which involves: 1) developing a point-cloud detector that can learn a general representation for localizing various objects, and 2) connecting textual and point-cloud representations to enable the detector to classify novel object categories based on text prompting. Specifically, we resort to rich image pre-trained models, by which the point-cloud detector learns localizing objects under the supervision of predicted 2D bounding boxes from 2D pre-trained detectors. Moreover, we propose a novel de-biased triplet cross-modal contrastive learning to connect the modalities of image, point-cloud and text, thereby enabling the point-cloud detector to benefit from vision-language pre-trained models, i.e., CLIP. The novel use of image and vision-language pretrained models for point-cloud detectors allows for openvocabulary 3D object detection without the need for 3D annotations. 

###  Training OV-3DET

------

#### Phase 1 
&emsp;Generating 3D pseudo box for training localization. You can generate pseudo-label of ScanNet by:

1. Prepare the [ScanNet](https://github.com/lyhdet/OV-3DET/blob/main/Prepare_ScanNet.md) dataset.
2. Moving:  "***domo.py***, ***scannet_pseudo_make.sh***, ***scannet_util.py*** " to the ***[Detic](https://github.com/facebookresearch/Detic)*** codebase.
3. Run the ***scannet_pseudo_make.sh***:    `bash scannet_pseudo_make.sh`
4. Replace the ground truth box of the training set with pseudo label.

 &emsp;

#### Phase 2 
&emsp;Connecting the embedding space of ***Text, Image and Point-cloud*** by running:  `bash scripts/scannet_quick_lr_7e-4.sh`


&emsp;
###  Evaluation

------

To evaluate OV-3DET, simply by running: `bash scripts/evaluate.sh`

&emsp;

### Acknowledgement

------

This codebase is modified base on ***3DETR*** [1], ***CLIP*** [2] and ***Detic*** [3], we sincerely appreciate their contributions!

[1] An end-to-end transformer model for 3d object detection. *ICCV*. 2021.

[2] Learning transferable visual models from natural language supervision. *ICML*. 2021.

[3] Detecting twenty-thousand classes using image-level supervision. *ECCV*. 2022.

&emsp;
### Citation

If you find this repository helpful, please consider citing our work:

```
@article{lu2023open,
  title={Open-Vocabulary Point-Cloud Object Detection without 3D Annotation},
  author={Lu, Yuheng and Xu, Chenfeng and Wei, Xiaobao and Xie, Xiaodong and Tomizuka, Masayoshi and Keutzer, Kurt and Zhang, Shanghang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
