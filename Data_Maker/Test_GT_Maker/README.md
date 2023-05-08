##  Generate GT Label for the Test Set
&emsp;Generating GT label for the test set. You can generate the ground truth of ScanNet by:
1. Prepare the [ScanNet](https://github.com/lyhdet/OV-3DET/blob/main/Phase%201/Prepare_ScanNet.md) dataset.
2. Moving:  "***domo.py***, ***scannet_pseudo_make.sh***, ***scannet_util.py*** " to the ***[Detic](https://github.com/facebookresearch/Detic)*** codebase.
3. Run the ***make_scannet_20cls_multi_thread.py***:
~~~
python make_scannet_20cls_multi_thread.py
~~~
