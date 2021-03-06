#!/bin/bash

TRAIN_DIR=./train/pspnet
DATASET_DIR=./datasets/
# CHECKPOINT_PATH=./train/pretrained/pspnet_v1_50.ckpt
# CHECKPOINT_EXCLUDE_SCOPES=global_step:0,pspnet_v1_50/pyramid_pool_module,pspnet_v1_50/fc1,pspnet_v1_50/logits
CHECKPOINT_EXCLUDE_SCOPES=global_step:0

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 train_semantic_segmentation.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_exclude_scopes=${CHECKPOINT_EXCLUDE_SCOPES} \
  --dataset_name=satelite \
  --dataset_split_name=train \
  --model_name=pspnet_v1_50 \
  --optimizer=momentum \
  --weight_decay=0.0001 \
	--learning_rate=0.01 \
  --learning_rate_decay_type=polynomial \
  --learning_rate_decay_factor=0.9 \
	--max_number_of_steps=150000 \
  --train_image_size=512 \
  --batch_size=4
