import argparse
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data/raw/train.txt"
DATASET_NAME = "opencsg/chinese-fineweb-edu"

CONTROL_CHAR_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]")
SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize web text into a stable plain-text format."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def iter_clean_samples(limit: Optional[int]):
    """Stream the dataset and yield cleaned text samples."""
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    for row in tqdm(ds, desc="Streaming chinese-fineweb-edu", unit="doc"):
        text = clean_text(row.get("text", ""))
        if len(text) < 20:
            continue
        yield text


def write_corpus(output_path: Path, limit: Optional[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_docs = 0
    written_chars = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for text in iter_clean_samples(limit=limit):
            fout.write(text)
            fout.write("\n\n")

            written_docs += 1
            written_chars += len(text)

            if limit is not None and written_docs >= limit:
                break

    print(f"Saved {written_docs} documents to {output_path}")
    print(f"Total characters: {written_chars}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download opencsg/chinese-fineweb-edu and convert it into a cleaned plain-text corpus."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output plain-text file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of documents to write.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    write_corpus(
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
