"""
Configurable Transformer block to dissect Lyapunov spectrum staircase.

A block computes:  x -> y
with optional skip connection, FFN, LN placement (pre/post/none).

We deliberately keep it small and deterministic so that Jacobian-based
Lyapunov computation is tractable.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BlockConfig:
    d_model: int = 16
    n_heads: int = 2
    d_ff: int = 32
    seq_len: int = 8
    use_skip: bool = True       # residual connection
    use_ffn: bool = True        # include FFN sublayer
    ffn_position: str = "after" # "after" attn, or "before" attn
    ln_style: str = "pre"       # "pre", "post", or "none"
    attn_only: bool = False     # if True, FFN disabled regardless


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)              # (B, T, H, dh)
        q = q.transpose(1, 2)                    # (B, H, T, dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = F.softmax(att, dim=-1)
        out = att @ v                            # (B, H, T, dh)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads)
        self.ffn = FFN(cfg.d_model, cfg.d_ff) if (cfg.use_ffn and not cfg.attn_only) else None
        if cfg.ln_style != "none":
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.ln2 = nn.LayerNorm(cfg.d_model) if self.ffn is not None else None
        else:
            self.ln1 = self.ln2 = None

    def _sub(self, x, sublayer, ln):
        """Apply a sublayer with chosen LN/skip combination."""
        if self.cfg.ln_style == "pre":
            y = sublayer(ln(x) if ln is not None else x)
            return x + y if self.cfg.use_skip else y
        elif self.cfg.ln_style == "post":
            y = sublayer(x)
            out = x + y if self.cfg.use_skip else y
            return ln(out) if ln is not None else out
        else:  # none
            y = sublayer(x)
            return x + y if self.cfg.use_skip else y

    def forward(self, x):
        if self.cfg.ffn_position == "before" and self.ffn is not None:
            x = self._sub(x, self.ffn, self.ln2)
            x = self._sub(x, self.attn, self.ln1)
        else:
            x = self._sub(x, self.attn, self.ln1)
            if self.ffn is not None:
                x = self._sub(x, self.ffn, self.ln2)
        return x


class TinyTransformer(nn.Module):
    """Stack of identical blocks. Used as the dynamical system x_{l+1}=F_l(x_l)."""
    def __init__(self, cfg: BlockConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layers)])

    def forward(self, x, return_states=False):
        states = [x] if return_states else None
        for blk in self.blocks:
            x = blk(x)
            if return_states:
                states.append(x)
        return (x, states) if return_states else x
