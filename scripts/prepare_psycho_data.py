import argparse
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data/psycho/markdown"
DEFAULT_OUTPUT = ROOT / "data/raw/psycho_train.txt"
DEFAULT_LLM_CACHE = ROOT / "data/psycho/llm_clean"
DEFAULT_API_BASE = "http://127.0.0.1:1137/v1"
DEFAULT_MODEL = "Jan-v3.5-4B-Q4_K_XL"

LLM_SYSTEM_PROMPT = (
    "你负责清洗扫描版PDF转出的书籍文本。你的输出将被逐字对照检查，必须严格保留原文全部内容与段落顺序，只允许做以下处理：\n"
    "1. 删除广告水印、网址、下载站宣传、版权声明行和孤立页码行；\n"
    "2. 修复明显的OCR错字、乱码符号（如 \" $#、口口、◆◆）和错误的全半角标点；\n"
    "3. 把被硬换行拆断的句子合并为完整段落。\n"
    "禁止总结、改写、扩写、续写、翻译、注释。无法确定的地方保留原样。输出长度应与输入基本一致（允许因删除水印略短）。直接输出清洗后的正文纯文本。"
)

CONTROL_CHAR_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]")
SPACE_RE = re.compile(r"[ \t]+")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
AUTOLINK_RE = re.compile(r"<(?:https?://|mailto:)[^>]+>")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def clean_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = CONTROL_CHAR_RE.sub("", line)
    line = IMAGE_RE.sub(r"\1", line)
    line = LINK_RE.sub(r"\1", line)
    line = AUTOLINK_RE.sub("", line)
    line = HTML_TAG_RE.sub("", line)
    line = HEADING_RE.sub("", line)
    line = BLOCKQUOTE_RE.sub("", line)
    line = LIST_MARKER_RE.sub("", line)

    if TABLE_SEPARATOR_RE.fullmatch(line):
        return ""
    if "|" in line:
        line = " ".join(part.strip() for part in line.strip(" |").split("|") if part.strip())

    line = line.replace("**", "").replace("__", "").replace("`", "")
    return SPACE_RE.sub(" ", line).strip()


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = []
    previous_line: Optional[str] = None
    in_fence = False

    for raw_line in text.splitlines():
        if FENCE_RE.match(raw_line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        line = clean_line(raw_line)
        if not line:
            continue
        if line == previous_line:
            continue

        cleaned_lines.append(line)
        previous_line = line

    return "\n".join(cleaned_lines).strip()


def iter_markdown_files(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.md")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(input_dir).parts)
    )


def chunk_paragraphs(text: str, chunk_chars: int) -> list[str]:
    """Pack paragraph blocks into chunks of at most `chunk_chars` characters."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for block in text.split("\n"):
        block_size = len(block) + 1
        if current and size + block_size > chunk_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(block)
        size += block_size
    if current:
        chunks.append("\n".join(current))
    return chunks


def chat_completion(api_base: str, model: str, user: str, max_tokens: int, timeout: int) -> tuple[str, str]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            choice = data["choices"][0]
            return choice["message"]["content"].strip(), choice.get("finish_reason", "stop")
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"chat completion failed after retries: {last_error}")


PREAMBLE_RE = re.compile(r"^(以下是|好的|抱歉|清洗后|这是|作为)")
WHITESPACE_RE = re.compile(r"\s+")


def bigram_overlap(output: str, reference: str) -> float:
    """Fraction of output character bigrams that also appear in the reference."""
    compact_output = WHITESPACE_RE.sub("", output)
    compact_reference = WHITESPACE_RE.sub("", reference)
    if len(compact_output) < 2:
        return 0.0
    output_bigrams = {compact_output[i : i + 2] for i in range(len(compact_output) - 1)}
    reference_bigrams = {compact_reference[i : i + 2] for i in range(len(compact_reference) - 1)}
    if not output_bigrams:
        return 0.0
    return len(output_bigrams & reference_bigrams) / len(output_bigrams)


def llm_clean_chunk(chunk: str, args: argparse.Namespace) -> tuple[str, str]:
    """Return (cleaned_text, status). Status: ok | fallback."""
    try:
        cleaned, finish_reason = chat_completion(
            api_base=args.api_base,
            model=args.model,
            user=chunk,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except RuntimeError as exc:
        print(f"\n[llm] request failed, keeping rule-cleaned text: {exc}")
        return chunk, "fallback"

    if finish_reason == "length":
        print("\n[llm] output truncated (finish_reason=length), keeping rule-cleaned text")
        return chunk, "fallback"
    if not cleaned or len(cleaned) < 0.5 * len(chunk):
        print("\n[llm] output suspiciously short, keeping rule-cleaned text")
        return chunk, "fallback"
    if PREAMBLE_RE.match(cleaned):
        print("\n[llm] output starts with preamble, keeping rule-cleaned text")
        return chunk, "fallback"

    overlap = bigram_overlap(cleaned, chunk)
    if overlap < args.min_overlap:
        print(f"\n[llm] fidelity check failed (bigram overlap {overlap:.2f} < {args.min_overlap}), keeping rule-cleaned text")
        return chunk, "fallback"
    return cleaned, "ok"


def write_llm_corpus(args: argparse.Namespace) -> None:
    files = list(iter_markdown_files(args.input_dir))
    if args.books:
        files = [path for path in files if any(token in path.stem for token in args.books)]
    if not files:
        raise FileNotFoundError(f"No Markdown files matched in {args.input_dir}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for path in files:
        cache_file = args.cache_dir / f"{path.stem}.txt"
        if cache_file.exists() and not args.force:
            print(f"[cache] {path.stem}: already cleaned, skipping")
            continue

        text = clean_markdown(path.read_text(encoding="utf-8", errors="ignore"))
        if len(text) < args.min_chars:
            print(f"[skip] {path.stem}: {len(text)} chars < {args.min_chars}")
            continue

        chunks = chunk_paragraphs(text, args.chunk_chars)
        if args.max_chunks is not None:
            chunks = chunks[: args.max_chunks]

        started = time.time()
        results: list[Optional[str]] = [None] * len(chunks)
        fallbacks = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(llm_clean_chunk, chunk, args): index
                for index, chunk in enumerate(chunks)
            }
            progress = tqdm(as_completed(futures), total=len(futures), desc=path.stem[:24], unit="chunk")
            for future in progress:
                index = futures[future]
                results[index], status = future.result()
                if status == "fallback":
                    fallbacks += 1

        cleaned_text = "\n".join(result for result in results if result)
        temp_file = cache_file.with_suffix(".tmp")
        temp_file.write_text(cleaned_text + "\n", encoding="utf-8")
        temp_file.replace(cache_file)

        elapsed = time.time() - started
        input_chars = sum(len(chunk) for chunk in chunks)
        print(
            f"[done] {path.stem}: {len(chunks)} chunks in {elapsed:.0f}s "
            f"({elapsed / max(len(chunks), 1):.1f}s/chunk), {fallbacks} fallbacks, "
            f"{input_chars} -> {len(cleaned_text)} chars"
        )

    assemble_llm_cache(args)


def assemble_llm_cache(args: argparse.Namespace) -> None:
    cache_files = sorted(args.cache_dir.glob("*.txt"))
    if not cache_files:
        print(f"No cleaned files in {args.cache_dir} yet")
        return

    written_docs = 0
    written_chars = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for cache_file in cache_files:
            text = cache_file.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) < args.min_chars:
                continue
            if not args.no_titles:
                fout.write(unicodedata.normalize("NFKC", cache_file.stem).strip())
                fout.write("\n")
            fout.write(text)
            fout.write("\n\n")
            written_docs += 1
            written_chars += len(text)

    print(f"Documents written: {written_docs}")
    print(f"Characters written: {written_chars}")
    print(f"Saved to: {args.output}")


def write_corpus(
    input_dir: Path,
    output_path: Path,
    min_chars: int,
    limit: Optional[int],
    include_titles: bool,
) -> None:
    files = list(iter_markdown_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No Markdown files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_docs = 0
    skipped_docs = 0
    written_chars = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for path in files:
            text = clean_markdown(path.read_text(encoding="utf-8", errors="ignore"))
            if len(text) < min_chars:
                skipped_docs += 1
                continue

            if include_titles:
                fout.write(unicodedata.normalize("NFKC", path.stem).strip())
                fout.write("\n")
            fout.write(text)
            fout.write("\n\n")

            written_docs += 1
            written_chars += len(text)
            if limit is not None and written_docs >= limit:
                break

    print(f"Markdown files found: {len(files)}")
    print(f"Documents written: {written_docs}")
    print(f"Documents skipped (< {min_chars} chars): {skipped_docs}")
    print(f"Characters written: {written_chars}")
    print(f"Saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean psychology Markdown files and combine them into a plain-text corpus."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing converted Markdown files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output plain-text corpus path.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip documents shorter than this after cleaning.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of documents to write.",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Do not prepend each document with its source filename.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Route each rule-cleaned chunk through a local LLM for OCR/watermark cleanup (cached per book).",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name served at --api-base.")
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1200,
        help="Max characters per LLM request (keep prompt+output within the server context).",
    )
    parser.add_argument("--max-tokens", type=int, default=2500, help="Generation cap per LLM request.")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel LLM requests per book.")
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.88,
        help="Min fraction of output character bigrams that must exist in the input chunk (anti-fabrication).",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--books",
        nargs="+",
        default=None,
        help="Substring filters; only process Markdown files whose stem contains one of them.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap chunks per book (smoke testing only; truncates the cached output).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_LLM_CACHE,
        help="Per-book cleaned text cache (enables resume across runs).",
    )
    parser.add_argument("--force", action="store_true", help="Re-clean books even if cached output exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_chars < 0:
        raise ValueError("--min-chars must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max-chunks must be positive")

    if args.llm:
        write_llm_corpus(args)
        return

    write_corpus(
        input_dir=args.input_dir,
        output_path=args.output,
        min_chars=args.min_chars,
        limit=args.limit,
        include_titles=not args.no_titles,
    )


if __name__ == "__main__":
    main()
