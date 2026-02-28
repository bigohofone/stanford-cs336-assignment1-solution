base_bsz=128

bsz=32
lr=0.0005

_lr=$(python -c "import math; print(round($lr * math.sqrt($bsz / $base_bsz), 6))")

result_path="./results/valid.json"

echo "Running: Trained on bsz=$bsz, lr=$_lr"

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_valid_model.py \
    training.lr=$_lr \
    training.batch_size=$bsz \
    paths.ckpt_id=final \
    paths.result_path=$result_path

echo "Completed!"