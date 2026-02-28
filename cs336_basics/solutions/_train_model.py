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


EMA_BETA = 0.9
WARMUP_RATIO = 0.1


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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    
    rank, wsz = setup_dist(config)
    is_main = (rank == 0)
    
    global_bsz = config.training.batch_size
    local_bsz = global_bsz // wsz
    
    dataset = load_dataset(config.paths.train_dataset_dir)
    loss_fn = CrossEntropyLoss(z=config.training.z)
    scheduler = get_lr_cosine_schedule if config.training.scheduler == "cosine" else get_lr_warmup_schedule
    
    model = TransformerLM(**OmegaConf.to_container(config.model, resolve=True))
    optimizer = AdamW(params=model.parameters(), lr=config.training.lr)
    
    ds_config = OmegaConf.to_container(config.deepspeed, resolve=True)
    ds_config.update({
        "train_batch_size": global_bsz,
        "train_micro_batch_size_per_gpu": local_bsz,
        "gradient_accumulation_steps": 1,
    })
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )
    load_path, client_state = model_engine.load_checkpoint(config.paths.ckpt_dir, tag=config.paths.get('ckpt_id', None))

    project_name, run_name = "cs336-a1", f"{config.run_name}"
    if is_main:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project=project_name, name=run_name, config=OmegaConf.to_container(config, resolve=True), resume="allow")

    start_iter = client_state['step'] + 1 if load_path else 0
    total_iters = config.training.total_iters
    warmup_iters = config.training.warmup_iters if config.training.warmup_iters is not None else int(total_iters * WARMUP_RATIO)
    
    pbar = tqdm(range(start_iter, total_iters), disable=(rank != 0), desc="Training")
    ema_loss = None
    if is_main:
        logger.info(f"Starting training from {start_iter} for {total_iters} iterations with global batch size {global_bsz} ({local_bsz} per GPU)")
        
    for step in pbar:
        current_lr = scheduler(step, config.training.lr, config.training.min_lr, warmup_iters, total_iters-warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        x, y = get_batch(dataset, local_bsz, config.training.max_seq_len, str(model_engine.device))
        outputs = model_engine(x)
        loss = loss_fn(outputs, y)
        model_engine.backward(loss)
        model_engine.step()
        
        if config.training.get('target_loss'):
            dist_loss = loss.clone().detach()
            deepspeed.comm.all_reduce(dist_loss, op=deepspeed.comm.ReduceOp.SUM)
            avg_loss = dist_loss.item() / wsz
            ema_loss = EMA_BETA * ema_loss + (1 - EMA_BETA) * avg_loss if ema_loss is not None else avg_loss
            if ema_loss <= config.training.target_loss and is_main:
                logger.info(f"\nTarget loss reached at step {step}")
                break

        if is_main:
            wandb.log({
                "train/step": step, "train/loss": loss.item(), 
                "train/lr": current_lr, "train/grad_norm": model_engine.get_global_grad_norm()
            })
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        if config.training.save_ckpt and step > 0 and step % config.training.save_interval == 0:
            client_state = {'step': step}
            model_engine.save_checkpoint(config.paths.ckpt_dir, tag=f"step_{step}", client_state=client_state)
            if is_main:
                logger.info(f"Checkpoint saved at step {step}.")

    if config.training.save_ckpt:
        client_state = {'step': config.training.total_iters}
        model_engine.save_checkpoint(config.paths.ckpt_dir, tag="final", client_state=client_state)
        if is_main:
            logger.info(f"Final checkpoint saved at step {config.training.total_iters}.")
    
    if is_main and config.paths.result_path:
        result_data = {
            "model_name": config.model.name,
            "learning_rate": config.training.lr,
            "batch_size": global_bsz,
            "total_iters": step,
            "final_loss": ema_loss if ema_loss is not None else loss.item(),
            "run_name": config.run_name,
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