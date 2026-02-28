import torch
import torch.nn as nn
import torch.optim as optim

import math

        
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    
    if it <= warmup_iters:
        lr = max_learning_rate * it / warmup_iters
    elif warmup_iters < it <= cosine_cycle_iters:
        freq = cosine_cycle_iters - warmup_iters
        lr = min_learning_rate + (1/2)*(max_learning_rate - min_learning_rate) * (1+math.cos(math.pi*(it-warmup_iters)/freq))
    else:
        lr = min_learning_rate
        
    return lr


def get_lr_warmup_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    total_iters: int,
) -> float:
    
    if it <= warmup_iters:
        lr = max_learning_rate * it / warmup_iters
    else:
        lr = max_learning_rate
        
    return lr