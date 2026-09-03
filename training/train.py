import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset import TextDataset
from model.config import model_config, train_config
from model.transformer import MiniLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniLLM on a single GPU server.")
    parser.add_argument(
        "--train-bin",
        type=Path,
        default=ROOT / "data/processed/train.bin",
        help="Path to the tokenized uint32 corpus.",
    )
    parser.add_argument(
        "--device",
        default=train_config.device,
        help="Training device, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=train_config.batch_size,
        help="Per-step micro batch size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=train_config.grad_accum_steps,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config.epochs,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=train_config.learning_rate,
        help="Peak learning rate after warmup.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=train_config.warmup_steps,
        help="Learning-rate warmup steps.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=train_config.weight_decay,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=train_config.grad_clip,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=train_config.num_workers,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=train_config.log_interval,
        help="Print logs every N optimizer steps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / train_config.checkpoint_dir,
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on optimizer steps, useful for smoke tests.",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=None,
        help="Save a checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint path.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pick_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, global_step, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "model_config": vars(model_config),
            "train_args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(train_config.seed)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dataset = TextDataset(str(args.train_bin), model_config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    raw_model = MiniLLM(model_config).to(device)
    model = raw_model
    if train_config.compile_model and not args.no_compile and hasattr(torch, "compile"):
        model = torch.compile(raw_model)

    fused_ok = device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=fused_ok,
    )

    updates_per_epoch = math.ceil(len(loader) / args.grad_accum_steps)
    total_updates = updates_per_epoch * args.epochs
    if args.max_steps is not None:
        total_updates = min(total_updates, args.max_steps)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_updates)

    amp_dtype = pick_amp_dtype(device)
    scaler = None
    if amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    global_step = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        print(f"Resumed from {args.resume} at epoch={start_epoch} step={global_step}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    run_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_updates = 0

        for step, (x, y) in enumerate(loader):
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")

            autocast_enabled = amp_dtype is not None
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum_steps != 0:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            epoch_updates += 1
            epoch_loss += loss.item() * args.grad_accum_steps

            if global_step % args.log_interval == 0:
                elapsed = time.time() - run_start
                samples_seen = global_step * args.batch_size * args.grad_accum_steps
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={epoch_loss / max(1, epoch_updates):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.6e} "
                    f"samples={samples_seen} elapsed_s={elapsed:.1f}"
                )

            if args.save_every_steps is not None and global_step % args.save_every_steps == 0:
                save_checkpoint(args.checkpoint_dir / "latest.pt", model, optimizer, scheduler, epoch, global_step, args)
                print(f"Saved checkpoint at step={global_step}")

        checkpoint_path = args.checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, global_step, args)
        save_checkpoint(args.checkpoint_dir / "latest.pt", model, optimizer, scheduler, epoch, global_step, args)
        print(f"Saved checkpoint to {checkpoint_path}")

        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
