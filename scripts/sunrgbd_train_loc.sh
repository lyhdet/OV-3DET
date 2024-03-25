#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--phase train_loc \
--dataset_name sunrgbd \
--clip_model ViT-L/14@336px \
--max_epoch 800 \
--nqueries 128 \
--base_lr 4e-4 \
--matcher_giou_cost 3 \
--matcher_cls_cost 0 \
--matcher_center_cost 5 \
--matcher_objectness_cost 0 \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir outputs/ov_3det_sunrgbd \
--ngpus 4 \
--dataset_num_workers 2 \
--dist_url tcp://localhost:32456 \
--batchsize_per_gpu 12
