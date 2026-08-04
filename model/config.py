from dataclasses import dataclass


@dataclass
class ModelConfig:

    vocab_size: int = 32000

    max_seq_len: int = 512

    n_layer: int = 12

    n_head: int = 12

    hidden_dim: int = 768

    dropout: float = 0.1



@dataclass
class TrainConfig:

    batch_size: int = 16

    learning_rate: float = 3e-4

    epochs: int = 10

    warmup_steps: int = 1000

    grad_clip: float = 1.0

    device: str = _DEFAULT_DEVICE


model_config = ModelConfig()

train_config = TrainConfig()