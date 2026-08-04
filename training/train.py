import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from model.transformer import MiniLLM
from model.config import *
from dataset import TextDataset


device = train_config.device


model = MiniLLM(model_config).to(device)


dataset = TextDataset(
    str(ROOT / "data/processed/train.bin"),
    model_config.max_seq_len,
)


loader = DataLoader(
    dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
)


for epoch in range(train_config.epochs):

    for step, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.grad_clip,
        )

        optimizer.step()

        if step % 100 == 0:

            print(
                epoch,
                step,
                loss.item(),
            )

    torch.save(
        model.state_dict(),
        str(ROOT / f"checkpoint_epoch{epoch}.pt"),
    )
