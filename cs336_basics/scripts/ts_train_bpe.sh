python -m cs336_basics.solutions._train_bpe \
    --config_path ./cs336_basics/configs/config.yml \
    --input_path ./data/TinyStoriesV2-GPT4-train.txt \
    --vocab_path ./out/ts_train/tokenizer/vocab.pkl \
    --merges_path ./out/ts_train/tokenizer/merges.pkl \
    --n_proc 8