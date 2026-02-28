mkdir -p results

base_bsz=128

lr=0.0005
total_iters=10000
result_path="./results/sweep_batch_results.json"
target_loss=1.7

batch_sizes="16 32 64 128 256 512"

for bsz in $batch_sizes; do
    _lr=$(python -c "import math; print(round($lr * math.sqrt($bsz / $base_bsz), 6))")
    _total_iters=$(python -c "import math; print(int($total_iters * $base_bsz / $bsz))")

    echo "Running Sweep: bsz=$bsz, lr=$_lr"

    uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
        training.lr=$_lr \
        training.batch_size=$bsz \
        training.total_iters=$_total_iters \
        training.save_ckpt=false \
        training.scheduler="constant" \
        paths.result_path=$result_path \
        training.target_loss=$target_loss
done

echo "All optimized sweeps completed!"