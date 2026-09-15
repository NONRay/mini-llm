import argparse
import json
import sys
import time
import urllib.request
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


def _debug_event(hypothesis_id: str, location: str, msg: str, data: dict):
    # #region debug-point shared:infer-debug
    _p = ROOT / ".dbg" / "sft-train-infer.env"
    _u = "http://127.0.0.1:7777/event"
    _s = "sft-train-infer"
    try:
        content = _p.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                _u = line.split("=", 1)[1]
            elif line.startswith("DEBUG_SESSION_ID="):
                _s = line.split("=", 1)[1]
        payload = {
            "sessionId": _s,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {msg}",
            "data": data,
            "ts": int(time.time() * 1000),
        }
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1,
        ).read()
    except Exception:
        pass
    # #endregion


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with MiniLLM.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="请简要介绍一下人工智能。",
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
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to previously generated token logits; 1.0 disables it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    prompt = args.prompt
    if "系统：" not in prompt and "用户：" not in prompt and "助手：" not in prompt:
        prompt = (
            "系统：你是一个认真、准确、简洁的中文助手。\n\n"
            f"用户：{prompt}\n\n"
            "助手："
        )

    tokens = tokenizer.encode(prompt, out_type=int)
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    # #region debug-point B:prompt-shape
    _debug_event(
        "B",
        "inference.py:prompt",
        "inference prompt encoded",
        {
            "prompt": prompt,
            "prompt_len": len(tokens),
            "has_system_tag": "系统：" in prompt,
            "has_user_tag": "用户：" in prompt,
            "has_assistant_tag": "助手：" in prompt,
        },
    )
    # #endregion

    print(prompt, end="", flush=True)

    for _ in range(args.max_tokens):
        with torch.no_grad():
            logits, _ = model(x)
        logits = logits[:, -1, :] / max(args.temperature, 1e-8)

        if args.repetition_penalty != 1.0:
            for token_id in set(x[0].tolist()):
                token_logit = logits[0, token_id]
                if token_logit < 0:
                    logits[0, token_id] = token_logit * args.repetition_penalty
                else:
                    logits[0, token_id] = token_logit / args.repetition_penalty

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
        if tokenizer.eos_id() >= 0 and int(next_token[0, 0].item()) == tokenizer.eos_id():
            break
        # #region debug-point D:sampling-steps
        if x.size(1) <= len(tokens) + 5:
            _debug_event(
                "D",
                "inference.py:sampling",
                "sampled next token",
                {
                    "generated_step": x.size(1) - len(tokens),
                    "token_id": int(next_token[0, 0].item()),
                    "token_text": token_str,
                },
            )
        # #endregion
        print(token_str, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
