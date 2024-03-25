python main.py \
--dataset_name sunrgbd \
--clip_model ViT-L/14@336px \
--nqueries 128 \
--test_ckpt outputs/ov_3det_sunrgbd/checkpoint.pth \
--test_only \
--dataset_num_workers 2 \
--batchsize_per_gpu 10 \
