import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import sentencepiece as spm

from model.transformer import MiniLLM
from model.config import *


device = train_config.device


tokenizer = spm.SentencePieceProcessor(
    model_file=str(ROOT / "data/tokenizer/tokenizer.model"),
)


model = MiniLLM(model_config).to(device)


model.load_state_dict(
    torch.load(
        str(ROOT / "checkpoint_epoch9.pt"),
        map_location=device,
    )
)


model.eval()


text = "Artificial intelligence"


tokens = tokenizer.encode(
    text,
    out_type=int,
)


x = torch.tensor(tokens).unsqueeze(0).to(device)


for _ in range(50):

    logits, _ = model(x)

    logits = logits[:, -1, :]

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    next = torch.multinomial(
        probs,
        1,
    )

    x = torch.cat(
        [x, next],
        dim=1,
    )


output = tokenizer.decode(
    x[0].tolist()
)


print(output)
