import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal
from jaxtyping import Bool, Float, Int

from .embedding import Embedding
from .transformer_block import TrasformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,
        device = None,
        dtype = None,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: Literal["silu", "swiglu"] | None = None,
        **kwargs
    ):
        super().__init__()
        self.use_post_norm = use_post_norm
        
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TrasformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype, 
                            remove_rmsnorm=remove_rmsnorm, use_post_norm=use_post_norm, remove_rope=remove_rope, ffn_type=ffn_type)
            for _ in range(num_layers)            
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if not remove_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(
        self, 
        x: Float[Tensor, "batch sequence_length d_model"]
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        
        if x.shape[1] > self.context_length:
            raise ValueError(
                f"Sequence length validation failed: Got input tensor with dim[1]={x.shape[1]}, "
                f"but model was initialized with context_length={self.context_length}. "
            )
            
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        
        if not self.use_post_norm:
            x = self.ln_final(x)
        out = self.lm_head(x)
        
        return out