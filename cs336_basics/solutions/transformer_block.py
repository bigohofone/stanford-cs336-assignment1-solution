import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal
from jaxtyping import Bool, Float, Int

from .multihead_self_attention import MultiHeadSelfAttentionWithRoPE, MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .positionwise_feedforward import SwiGLU, SiLU

class TrasformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device = None,
        dtype = None,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: Literal["silu", "swiglu"] | None = None
    ):
        super().__init__()
        attn_type = MultiHeadSelfAttentionWithRoPE if not remove_rope else MultiHeadSelfAttention
        self.attn = attn_type(
            d_model=d_model, num_heads=num_heads,
            max_seq_len=max_seq_len, theta=theta,
            device=device, dtype=dtype
        )
        
        norm_type = RMSNorm if not remove_rmsnorm else nn.Identity
        self.ln1 = norm_type(d_model, device=device, dtype=dtype) 
        self.ln2 = norm_type(d_model, device=device, dtype=dtype) 
        
        ffn_type = SiLU if ffn_type=='silu' else SwiGLU
        self.ffn = ffn_type(
            d_model=d_model, d_ff=d_ff, 
            device=device, dtype=dtype
        )
        
        self.use_post_norm = use_post_norm
        
    def forward(
        self, 
        x: Float[Tensor, "batch sequence_length d_model"]
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        if not self.use_post_norm:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        
        return x