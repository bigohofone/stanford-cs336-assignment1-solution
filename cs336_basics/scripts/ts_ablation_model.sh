uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml \
    --update_config_path ./cs336_basics/configs/remove_rmsnorm.yml

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml \
    --update_config_path ./cs336_basics/configs/remove_rope.yml

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml \
    --update_config_path ./cs336_basics/configs/use_post_norm.yml

uv run deepspeed --num_gpus=8 ./cs336_basics/solutions/_train_model.py \
    --base_config_path ./cs336_basics/configs/default.yml \
    --update_config_path ./cs336_basics/configs/use_silu.yml