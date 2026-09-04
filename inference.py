import argparse
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

checkpoint = torch.load(
    str(ROOT / "checkpoints/latest.pt"),
    map_location=device,
    weights_only=False,
)

state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
model.load_state_dict(state_dict)

model.eval()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with MiniLLM.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="人工智能是",
        help="Text prompt to start generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature; 0.0 means greedy (argmax).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling; 0 means no restriction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokens = tokenizer.encode(args.prompt, out_type=int)
    x = torch.tensor(tokens).unsqueeze(0).to(device)

    print(args.prompt, end="", flush=True)

    for _ in range(args.max_tokens):
        with torch.no_grad():
            logits, _ = model(x)
        logits = logits[:, -1, :] / max(args.temperature, 1e-8)

        if args.top_k > 0:
            top_vals, _ = torch.topk(logits, args.top_k)
            logits[logits < top_vals[:, [-1]]] = -float("inf")

        if args.temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        x = torch.cat([x, next_token], dim=1)

        token_str = tokenizer.decode(next_token[0].tolist())
        print(token_str, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
