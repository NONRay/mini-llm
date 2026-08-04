import os
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent

input_file = str(ROOT / "data/raw/train.txt")
model_prefix = str(ROOT / "data/tokenizer/tokenizer")
os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=model_config.vocab_size,
    model_type="bpe",
    character_coverage=1.0,
)
