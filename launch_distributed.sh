#!/bin/bash
# filepath: /home/hw3579/bart/launch_distributed.sh

# 设置可见的GPU
export CUDA_VISIBLE_DEVICES=2,3

# 检查GPU数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

# 启动分布式训练
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    --config distributed_2gpu.yaml

# 或者使用特定配置
# torchrun --standalone --nproc_per_node=2 train.py --config distributed_2gpu.yaml
# torchrun --standalone --nproc_per_node=4 train.py --config distributed_4gpu.yaml
# torchrun --standalone --nproc_per_node=8 train.py --config distributed_8gpu.yaml