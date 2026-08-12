from dataclasses import dataclass

import torch


_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class ModelConfig:
    """模型结构超参数。"""
    # 词表大小与上下文窗口长度
    vocab_size: int = 32000

    max_seq_len: int = 1024

    # Transformer 层数、注意力头数和隐藏维度
    n_layer: int = 8
    n_head: int = 8
    hidden_dim: int = 512

    dropout: float = 0.1


@dataclass
class TrainConfig:
    """训练过程超参数。"""
    batch_size: int = 16

    learning_rate: float = 3e-4

    epochs: int = 5

    warmup_steps: int = 1000

    grad_clip: float = 1.0

    weight_decay: float = 0.1

    device: str = _DEFAULT_DEVICE


model_config = ModelConfig()
train_config = TrainConfig()
