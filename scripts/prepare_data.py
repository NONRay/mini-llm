import os
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent

tokenizer_path = str(ROOT / "data/tokenizer/tokenizer.model")
bin_path = ROOT / "data/processed/train.bin"
os.makedirs(bin_path.parent, exist_ok=True)

sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

with open(ROOT / "data/raw/train.txt", encoding="utf8") as f:
    text = f.read()

tokens = sp.encode(text, out_type=int)
tokens = np.array(tokens, dtype=np.uint32)

m = np.memmap(str(bin_path), dtype=np.uint32, mode="w+", shape=tokens.shape)
m[:] = tokens
m.flush()

print("tokens:", len(tokens))
