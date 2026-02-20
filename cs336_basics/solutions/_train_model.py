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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by deepspeed launcher")
parser.add_argument("--base_config_path", type=str, help="Path to the YAML configuration file")
parser.add_argument('--update_config_path', type=str, nargs='*', help="Additional config updates in the form of key=value pairs")
parser.add_argument('--test', action='store_true', help="Whether to run in test mode with fewer iterations")
args = parser.parse_args()


with open(args.base_config_path, 'r') as f:
    config = yaml.safe_load(f)

for update_path in args.update_config_path or []:
    with open(update_path, 'r') as f:
        update = yaml.safe_load(f)
        for key, value in update.items():
            config[key] = value
            
if args.test:
    config['training']['total_iters'] = 10
    config['training']['warmup_iters'] = 5
    config['training']['save_interval'] = 5
    config['paths']['save_ckpt_dir'] = "./out/test_ckpts"
    config['wandb']['name'] = "test"


torch.cuda.set_device(args.local_rank)
deepspeed.init_distributed()
rank = deepspeed.comm.get_rank()
seed_everything(42 + rank)


if rank == 0:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=config['wandb']['project'], 
               name=f"run-{config['wandb']['name']}-{wandb.util.generate_id()}",
               config=vars(args), resume="allow")

full_dataset = load_dataset(config['paths']['dataset_dir'])

model = TransformerLM(**config['model'])
optimizer = AdamW(params=model.parameters(), lr=config['training']['lr'])

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=optimizer,
    config=config['deepspeed']
)

loss_fn = CrossEntropyLoss(z=config['training']['z'])

start_step = 0
if config['paths']['load_ckpt_path'] is not None:
    _, client_state = model_engine.load_checkpoint(config['paths']['load_ckpt_path'])
    start_step = client_state.get('step', 0) + 1
    if rank == 0:
        print(f"Resuming training from step {start_step}")

pbar = tqdm(range(start_step, config['training']['total_iters']), disable=(rank != 0), desc="Training")

for step in pbar:
    current_lr = get_lr_cosine_schedule(
        it=step,
        max_learning_rate=config['training']['lr'],
        min_learning_rate=config['training']['min_lr'],
        warmup_iters=config['training']['warmup_iters'],
        cosine_cycle_iters=config['training']['total_iters']
    )

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    x_batch, y_batch = get_batch(
        dataset=full_dataset,
        batch_size=config['deepspeed']['train_micro_batch_size_per_gpu'],
        context_length=config['training']['max_seq_len'],
        device=str(model_engine.device)
    )

    outputs = model_engine(x_batch)
    loss = loss_fn(outputs, y_batch)

    model_engine.backward(loss)

    clip_grad_norm_(
        parameters=model_engine.parameters(),
        max_l2_norm=config['training']['max_grad_norm']
    )

    model_engine.step()

    if rank == 0:
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": current_lr,
            "train/step": step
        })
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

    if step > 0 and step % config['training']['save_interval'] == 0:
        client_state = {'step': step}
        model_engine.save_checkpoint(config['paths']['save_ckpt_dir'], tag=f"step_{step}", client_state=client_state)
        if rank == 0:
            pbar.write(f"[Step {step}] Checkpoint saved.")

client_state = {'step': config['training']['total_iters']}
model_engine.save_checkpoint(config['paths']['save_ckpt_dir'], tag="final", client_state=client_state)

if rank == 0:
    print("Training finished.")
    wandb.finish()