import os
import json
import argparse
from .train_bpe import train_bpe, SPLIT_SPECIAL_TOKEN


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--vocab_path', type=str)
parser.add_argument('--merges_path', type=str)
parser.add_argument('--n_proc', type=int, default=4)
args = parser.parse_args()

with open(args.config_path, 'r') as f:
    import yaml
    config = yaml.safe_load(f)

output_dir = os.path.dirname(args.vocab_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

train_bpe(
    input_path=args.input_path,
    vocab_size=config['tokenizer']['vocab_size'],
    special_tokens=config['tokenizer']['special_tokens'],
    vocab_path=args.vocab_path,
    merges_path=args.merges_path,
    n_proc=args.n_proc
)
