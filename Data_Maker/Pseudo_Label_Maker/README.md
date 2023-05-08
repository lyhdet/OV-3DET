##  Generate Pseudo Label for the Training Set
&emsp;Generating 3D pseudo box for training localization. You can generate pseudo-label of ScanNet by:
1. Prepare the [ScanNet](https://github.com/lyhdet/OV-3DET/blob/main/Phase%201/Prepare_ScanNet.md) dataset.
2. Moving:  "***domo.py***, ***scannet_pseudo_make.sh***, ***scannet_util.py*** " to the ***[Detic](https://github.com/facebookresearch/Detic)*** codebase.
3. Run the ***scannet_pseudo_make.sh***:
~~~
bash scannet_pseudo_make.sh
~~~
