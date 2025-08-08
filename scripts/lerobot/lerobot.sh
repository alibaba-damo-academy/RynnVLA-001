#!/bin/bash

export NCCL_IB_HCA=mlx5 
export NCCL_IB_TC=136 
export NCCL_IB_SL=5 
export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_DEBUG=INFO 
export NCCL_IB_TIMEOUT=3600 
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 
export NCCL_LAUNCH_MODE=PARALLEL 
export TOKENIZERS_PARALLELISM=false

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-2}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "master_addr: $MASTER_ADDR"
echo "master_port: $MASTER_PORT"
echo "rank: $RANK"


torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    train.py \
    --config_file ./configs/lerobot/lerobot_exp.yml \
    --exp_dir './experiments/lerobot'
