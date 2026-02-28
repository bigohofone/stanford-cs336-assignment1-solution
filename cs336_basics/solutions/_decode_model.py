import torch
import torch.nn.functional as F
from cs336_basics.solutions.tokenizer import Tokenizer
from cs336_basics.solutions.transformer_lm import TransformerLM
from cs336_basics.solutions.decoding import decode
import os
import json
import glob
import random
import argparse
import numpy as np
import deepspeed
import wandb
import torch
import yaml
from tqdm import tqdm
from collections.abc import Mapping

from cs336_basics.solutions.transformer_lm import TransformerLM
from cs336_basics.solutions.adamw import AdamW
from cs336_basics.solutions.cross_entropy import CrossEntropyLoss
from cs336_basics.solutions.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.solutions.gradient_clipping import clip_grad_norm_
from cs336_basics.solutions.data_loading import get_batch


def load_dataset(dataset_dir):
    pattern = os.path.join(dataset_dir, "*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shard_w*.npy files found in {dataset_dir}")
    
    arrays = [np.load(f) for f in files]
    full_dataset = np.concatenate(arrays)
    return full_dataset

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by deepspeed launcher")
parser.add_argument("--config_path", type=str, help="Path to the YAML configuration file")
args = parser.parse_args()

ckpt_dir = '/home/aikusrv02/_PROJECTS/wonjunoh/assignment1-basics-solution/checkpoints/best'


with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
    
torch.cuda.set_device(args.local_rank)
deepspeed.init_distributed()
rank = deepspeed.comm.get_rank()
seed_everything(42 + rank)
    
model = TransformerLM(**config['model'])

validation_config = config['deepspeed'].copy()
validation_config["zero_optimization"]["stage"] = 0

model_engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=None,
    config=validation_config
)
load_path, _ = model_engine.load_checkpoint(config['paths']['ckpt_dir'], tag="final")
if load_path is None:
    raise ValueError(f"체크포인트를 로드하지 못했습니다: {config['paths']['ckpt_dir']}의 'final' 태그를 확인하세요.")

model_engine.eval()

tokenizer = Tokenizer.from_files(config['paths']['vocab_path'], config['paths']['merges_path'], **config['tokenizer'])

prompt = "Once upon a time"
generated_text = decode(model_engine, tokenizer, prompt)
print("Generated Text:")
print(generated_text)