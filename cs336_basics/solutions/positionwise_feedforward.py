import torch
import torch.nn as nn
import math

from .linear import Linear

class SiLU(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None, **kwargs):
        super().__init__()
        self.silu = SiLU()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        
    def forward(self, x: torch.Tensor):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class FFNSiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None, **kwargs):
        super().__init__()
        self.silu = SiLU()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        
    def forward(self, x: torch.Tensor):
        return self.w2(self.silu(self.w1(x)))