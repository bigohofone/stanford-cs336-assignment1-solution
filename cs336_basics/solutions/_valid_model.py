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
import torch.distributed as dist

from cs336_basics.solutions.transformer_lm import TransformerLM
from cs336_basics.solutions.adamw import AdamW
from cs336_basics.solutions.cross_entropy import CrossEntropyLoss
from cs336_basics.solutions.learning_rate_schedule import get_lr_cosine_schedule, get_lr_warmup_schedule
from cs336_basics.solutions.gradient_clipping import clip_grad_norm_
from cs336_basics.solutions.data_loading import get_batch


import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import logging

logger = logging.getLogger(__name__)


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


def setup_dist(config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    
    wsz = deepspeed.comm.get_world_size()
    rank = deepspeed.comm.get_rank()
    
    seed = config.training.get('seed', 42) + rank
    seed_everything(seed)
    
    return rank, wsz

@torch.no_grad()
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):

    rank, wsz = setup_dist(config)
    is_main = (rank == 0)
    
    global_bsz = config.validation.batch_size
    local_bsz = global_bsz // wsz
    
    valid_dataset = load_dataset(config.paths.valid_dataset_dir)
    loss_fn = CrossEntropyLoss(z=config.training.z)
    model = TransformerLM(**OmegaConf.to_container(config.model, resolve=True))
    
    ds_config = OmegaConf.to_container(config.deepspeed, resolve=True)
    ds_config.update({
        "train_batch_size": global_bsz,
        "train_micro_batch_size_per_gpu": local_bsz,
        "gradient_accumulation_steps": 1,
    })
    ds_config['zero_optimization']['stage'] = 0
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    
    load_path, client_state = model_engine.load_checkpoint(
        config.paths.ckpt_dir, 
        tag=config.paths.get('ckpt_id', None)
    )
    model_engine.eval()

    project_name, run_name = "cs336-a1", f"{config.run_name}_{config.paths.get('ckpt_id', None)}"
    if is_main:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=project_name, 
            name=run_name, 
            config=OmegaConf.to_container(config, resolve=True), 
            resume="allow"
        )

    pbar = tqdm(range(0, config.validation.total_iters), disable=(rank != 0), desc="Training")
    total_loss = torch.zeros(1).to(model_engine.device)
        
    for step in pbar:
        x, y = get_batch(valid_dataset, local_bsz, config.training.max_seq_len, str(model_engine.device))
        outputs = model_engine(x)
        loss = loss_fn(outputs, y)
        total_loss += loss.detach()

        if is_main:
            wandb.log({
                "valid/step": step, "valid/loss": loss.item(),
            })
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)   
    
    if is_main and config.paths.result_path:
        result_data = {
            "run_name": config.run_name,
            "train_iters": client_state['step'],
            "valid_loss": total_loss.item() / (wsz * len(pbar))
        }
        import pathlib
        result_path = pathlib.Path(config.paths.result_path)
        
        if result_path.exists():
            with open(config.paths.result_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
            
        existing_data.append(result_data)
        
        with open(config.paths.result_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        logger.info(f"Results saved to {config.paths.result_path}")
        
    if is_main:
        wandb.finish()
    
if __name__ == "__main__":
    import sys
    sys.argv = [arg for arg in sys.argv if "--local_rank" not in arg]
    main()