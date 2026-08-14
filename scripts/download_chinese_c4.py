import argparse
import io
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
import zstandard as zstd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data/raw/train.txt"
DATASET_NAME = "shjwudp/chinese-c4"

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


def iter_shard_paths(cache_dir: Path):
    files = list_repo_files(DATASET_NAME, repo_type="dataset")
    shards = sorted(file for file in files if file.startswith("data/") and file.endswith(".jsonl.zst"))
    for shard in shards:
        last_error = None
        for attempt in range(1, 4):
            try:
                yield Path(
                    hf_hub_download(
                        repo_id=DATASET_NAME,
                        filename=shard,
                        repo_type="dataset",
                        cache_dir=str(cache_dir),
                    )
                )
                break
            except Exception as exc:
                last_error = exc
                print(f"Download failed for {shard} (attempt {attempt}/3): {exc}")
                if attempt < 3:
                    time.sleep(2)
        else:
            raise RuntimeError(f"Failed to download shard after retries: {shard}") from last_error


def iter_clean_samples(cache_dir: Path):
    for shard_path in iter_shard_paths(cache_dir):
        with shard_path.open("rb") as fin:
            reader = zstd.ZstdDecompressor().stream_reader(fin)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                row = json.loads(line)
                text = clean_text(row.get("text", ""))
                if len(text) < 20:
                    continue
                yield text


def write_corpus(output_path: Path, limit: Optional[int], cache_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    written_docs = 0
    written_chars = 0

    with output_path.open("w", encoding="utf-8") as fout:
        progress = tqdm(desc="Cleaning chinese-c4", unit="doc")
        for text in iter_clean_samples(cache_dir=cache_dir):
            fout.write(text)
            fout.write("\n\n")

            written_docs += 1
            written_chars += len(text)
            progress.update(1)

            if limit is not None and written_docs >= limit:
                break

        progress.close()

    print(f"Saved {written_docs} documents to {output_path}")
    print(f"Total characters: {written_chars}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download shjwudp/chinese-c4 and convert it into a cleaned plain-text corpus."
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/cache/chinese_c4",
        help="Local cache directory for downloaded dataset shards.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    write_corpus(
        output_path=args.output,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
