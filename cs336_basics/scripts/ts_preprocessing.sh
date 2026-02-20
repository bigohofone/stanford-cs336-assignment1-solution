python -m cs336_basics.solutions._preprocessing \
    --config_path ./cs336_basics/configs/config.yml \
    --vocab_path ./out/ts_train/tokenizer/vocab.pkl \
    --merges_path ./out/ts_train/tokenizer/merges.pkl \
    --input_path ./data/TinyStoriesV2-GPT4-valid.txt \
    --output_path ./out/ts_valid/data.npy \
    --n_proc 8

python -m cs336_basics.solutions._preprocessing \
    --config_path ./cs336_basics/configs/config.yml \
    --vocab_path ./out/ts_train/tokenizer/vocab.pkl \
    --merges_path ./out/ts_train/tokenizer/merges.pkl \
    --input_path ./data/TinyStoriesV2-GPT4-train.txt \
    --output_path ./out/ts_train/data.npy \
    --n_proc 8