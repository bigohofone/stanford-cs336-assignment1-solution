mkdir -p results

bsz=128
total_iters=2000
result_path="./results/sweep_lr_results.json"

lrs=(0.01 0.005 0.001 0.0005 0.0001)

for lr in "${lrs[@]}"; do
    echo "Running Sweep: bsz=$bsz, lr=$lr"

    uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
        training.lr="$lr" \
        training.batch_size="$bsz" \
        training.total_iters="$total_iters" \
        training.save_ckpt=false \
        training.scheduler="constant" \
        paths.result_path="$result_path"
done

echo "All optimized sweeps completed!"