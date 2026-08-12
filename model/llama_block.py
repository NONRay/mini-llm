import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """RMSNorm：按最后一个维度归一化，计算开销低于 LayerNorm。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络，通过门控激活提升表达能力。"""

    def __init__(self, dim):
        super().__init__()
        hidden = dim * 4
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.w3 = nn.Linear(dim, hidden)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """一个 Pre-Norm Transformer 解码器模块。"""

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim)
        self.ffn = SwiGLU(config.hidden_dim)

    def forward(self, x, return_attention=False):
        if return_attention:
            attention_output, att = self.attn(self.norm1(x), return_attention=True)
            x = x + attention_output
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return (x, att) if return_attention else x
