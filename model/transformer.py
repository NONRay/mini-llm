import torch
import torch.nn as nn

from .llama_block import RMSNorm, TransformerBlock


class MiniLLM(nn.Module):
    """由 token/位置嵌入、多个 Transformer 块和语言模型头组成的小型语言模型。"""

    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # 权重绑定可减少参数量，并让输入输出词向量保持一致。
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None, return_hidden=False):
        _, seq_len = idx.shape
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        hidden = []
        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hidden.append(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return (logits, loss, hidden) if return_hidden else (logits, loss)
