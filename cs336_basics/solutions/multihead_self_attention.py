import torch
import torch.nn as nn
from einops import rearrange

from .rope import RoPE
from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, **kwargs):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
    
        self.qkv_proj = Linear(d_model, 3*d_model, device, dtype) 
        self.output_proj = Linear(d_model, d_model, device, dtype)
        
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        
        
    def forward(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        
        out = self.scaled_dot_product_attention(q, k, v, mask) # b h s d
        out = rearrange(out, "b h s d -> b s (h d)")
        
        out = self.output_proj(out)
        
        return out
        

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None, **kwargs):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
    
        self.d_head = self.d_model // num_heads
        
        self.qkv_proj = Linear(d_model, 3*d_model, device, dtype) 
        self.output_proj = Linear(d_model, d_model, device, dtype)
        
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        self.rope = RoPE(theta, self.d_head, max_seq_len, device, dtype)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        if not token_positions:
            token_positions = torch.arange(0, seq_len, device=x.device)
        
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        
        out = self.scaled_dot_product_attention(q, k, v, mask) # b h s d
        out = rearrange(out, "b h s d -> b s (h d)")
        
        out = self.output_proj(out)
        
        return out