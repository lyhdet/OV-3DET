# OV-3DET: Open-Vocabulary Point-Cloud Object Detection without 3D Annotation

Accepted to CVPR2023. &emsp;

###  Running OV-3DET

------

The phase 1 is to generate 3D pseudo box for training localization. You can generate pseudo-label of ScanNet by:

1. Prepare the ScanNet dataset.
2. Moving:  "***domo.py***, ***scannet_pseudo_make.sh***, ***scannet_util.py*** " to the ***Detic*** codebase.
3. Run the ***scannet_pseudo_make.sh***:    `bash scannet_pseudo_make.sh`
4. Replace the ground truth box of the training set with pseudo label.

 &emsp;

The phase 2 is to connect the embedding space of ***Text, Image and Point-cloud*** by running:  `bash scripts/scannet_quick_lr_7e-4.sh`



###  Evaluation

------

To evaluate OV-3DET, simply by running: `bash scripts/evaluate.sh`

&emsp;

### Acknowledgement

------

This codebase is modified base on ***3DETR*** [1], ***CLIP*** [2] and ***Detic*** [3], we sincerely appreciate their contributions!

&emsp;

[1] An end-to-end transformer model for 3d object detection. *ICCV*. 2021.

[2] Learning transferable visual models from natural language supervision. *ICML*. 2021.

[3] Detecting twenty-thousand classes using image-level supervision. *ECCV*. 2022.