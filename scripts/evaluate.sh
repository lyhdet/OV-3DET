python main.py \
--dataset_name scannet \
--nqueries 128 \
--test_ckpt outputs/ov_3det_scannet/checkpoint.pth \
--test_only \
--dataset_num_workers 2 \
--batchsize_per_gpu 10 \
