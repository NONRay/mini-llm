import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream raw text through SentencePiece and write uint32 tokens to train.bin."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/train.txt",
        help="Path to the raw text corpus.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "data/tokenizer/tokenizer.model",
        help="Path to the SentencePiece model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/train.bin",
        help="Path to the output uint32 binary.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100000,
        help="Update tqdm every N input lines.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    total_tokens = 0
    total_lines = 0

    with args.input.open("r", encoding="utf-8") as fin, args.output.open("wb") as fout:
        progress = tqdm(desc="Tokenizing corpus", unit="line")
        for line in fin:
            tokens = sp.encode(line, out_type=int)
            if tokens:
                np.asarray(tokens, dtype=np.uint32).tofile(fout)
                total_tokens += len(tokens)

            total_lines += 1
            if total_lines % args.progress_interval == 0:
                progress.update(args.progress_interval)

        remainder = total_lines % args.progress_interval
        if remainder:
            progress.update(remainder)
        progress.close()

    print(f"lines: {total_lines}")
    print(f"tokens: {total_tokens}")
    print(f"saved_to: {args.output}")


if __name__ == "__main__":
    main()
