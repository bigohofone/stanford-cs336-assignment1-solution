base_bsz=128
bsz=32
lr=0.0005
total_iters=10000

_lr=$(python -c "import math; print(round($lr * math.sqrt($bsz / $base_bsz), 6))")
_total_iters=$(python -c "import math; print(int($total_iters * $base_bsz / $bsz))")

echo "Running: bsz=$bsz, lr=$_lr"

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    training.lr=$_lr \
    training.batch_size=$bsz \
    training.total_iters=$_total_iters

echo "Completed!"