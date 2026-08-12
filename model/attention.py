import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """带因果遮罩的多头自注意力，保证当前位置不会看到未来 token。"""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.hidden_dim = config.hidden_dim
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x, return_attention=False):
        batch_size, seq_len, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_dim = channels // self.n_head

        def split_heads(value):
            return value.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        att = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = self.dropout(F.softmax(att, dim=-1))
        output = (att @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        output = self.proj(output)
        return (output, att) if return_attention else output
