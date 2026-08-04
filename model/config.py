from dataclasses import dataclass


@dataclass
class ModelConfig:

    # tokenizer
    vocab_size: int = 32000

    # context length
    max_seq_len: int = 1024

    # Transformer size
    n_layer: int = 8
    n_head: int = 8
    hidden_dim: int = 512

    dropout: float = 0.1


@dataclass
class TrainConfig:

    batch_size: int = 16

    learning_rate: float = 3e-4

    epochs: int = 5

    warmup_steps: int = 1000

    grad_clip: float = 1.0

    weight_decay: float = 0.1

    device: str = _DEFAULT_DEVICE


model_config = ModelConfig()

train_config = TrainConfig()