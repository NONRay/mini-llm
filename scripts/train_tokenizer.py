import argparse
from pathlib import Path

import sentencepiece as spm

from model.config import model_config


ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/train.txt",
        help="Path to the raw text corpus.",
    )
    parser.add_argument(
        "--model-prefix",
        type=Path,
        default=ROOT / "data/tokenizer/tokenizer",
        help="Output prefix for tokenizer.model/tokenizer.vocab.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=model_config.vocab_size,
        help="SentencePiece vocabulary size.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="SentencePiece worker threads.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(args.input),
        model_prefix=str(args.model_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        num_threads=args.num_threads,
    )

    print(f"Saved tokenizer to {args.model_prefix}.model")


if __name__ == "__main__":
    main()
