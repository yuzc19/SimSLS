#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

export CUDA_VISIBLE_DEVICES=3

# 38599MiB / 4 hours
NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Add gradient_accumulation_steps
# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
python train.py \
    --model_name_or_path thunlp/Lawformer \
    --train_file train/train.json \
    --output_dir result/lawformer \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --max_seq_length 3072 \
    --evaluation_strategy steps \
    --metric_for_best_model ndcg \
    --load_best_model_at_end \
    --eval_steps 50 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.1 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"