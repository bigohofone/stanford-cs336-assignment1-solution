#!/bin/bash

mkdir -p results

batch_size=128
lr="0.0005"
total_iters=2000
result_path="./results/sweep_model_results.json"

model_configs=(
    "use_post_norm"
    "remove_rope"
    "use_silu"
    "remove_rmsnorm"
)

for model_config in "${model_configs[@]}"; do
    echo "Running Sweep: model_config=$model_config"
    uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_sweep_hyperparameters.py \
        model=$model_config \
        training.lr=$_lr \
        training.batch_size=$bsz \
        training.total_iters=$_total_iters \
        paths.result_path=$result_path \

echo "All optimized sweeps completed!"