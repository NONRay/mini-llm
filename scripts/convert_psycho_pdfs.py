import argparse
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from fnmatch import fnmatch
from urllib.parse import unquote

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_DIR = ROOT / "data/psycho/pdf"
DEFAULT_OUT_DIR = ROOT / "data/psycho/markdown"

UNSAFE_CHARS_RE = re.compile(r'[\\/:*?"<>|]')
MULTI_SPACE_RE = re.compile(r"\s+")
IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

def sanitize_stem(stem: str) -> str:
    stem = unquote(stem)
    stem = UNSAFE_CHARS_RE.sub("_", stem)
    stem = MULTI_SPACE_RE.sub(" ", stem).strip(" .")
    return stem or "untitled"

def count_pages(pdf_path: Path) -> Optional[int]:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        pass
    try:
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdftypes import resolve1

        with pdf_path.open("rb") as handle:
            document = PDFDocument(PDFParser(handle))
            return int(resolve1(document.catalog["Pages"])["Count"])
    except Exception:
        return None

def looks_scanned(text: Optional[str], pages: Optional[int], min_chars_per_page: int) -> bool:
    if not text or len(text.strip()) < 200:
        return True
    if pages:
        return len(text) / pages < min_chars_per_page
    return False

def convert_with_markitdown(pdf_path: Path) -> Optional[str]:
    try:
        from markitdown import MarkItDown

        result = MarkItDown().convert(str(pdf_path))
        return result.text_content
    except Exception as exc:
        print(f"\n[markitdown] {pdf_path.name}: {exc}")
        return None

def find_mineru_binary() -> str:
    for name in ("mineru", "magic-pdf"):
        found = shutil.which(name)
        if found:
            return found
    return ""

def convert_with_mineru(
    binary: str, pdf_path: Path, tmp_dir: Path, extra_args: list, chunk_pages: int
) -> Optional[str]:
    if Path(binary).name != "mineru":
        cmd = [binary, "-p", str(pdf_path), "-o", str(tmp_dir), *extra_args]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\n[{binary}] {pdf_path.name}: exit code {exc.returncode}")
            return None
        candidates = list(tmp_dir.rglob("*.md"))
        if not candidates:
            print(f"\n[{binary}] {pdf_path.name}: no markdown output")
            return None
        best = max(candidates, key=lambda path: path.stat().st_size)
        return best.read_text(encoding="utf-8", errors="ignore")

    total = count_pages(pdf_path)
    if chunk_pages > 0 and not total:
        print(f"\n[mineru] {pdf_path.name}: cannot count pages; skipping to avoid full-book timeout", flush=True)
        return None
    ranges = (
        [(s, min(s + chunk_pages - 1, total)) for s in range(1, total + 1, chunk_pages)]
        if chunk_pages > 0 else []
    )
    attempts = len(ranges) if ranges else 1
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, bounds in enumerate(ranges or [None]):
        chunk_path = tmp_dir / f"chunk_{index:03d}.md"
        label = f"pages {bounds[0]}-{bounds[1]}" if bounds else "all pages"
        print(f"[mineru] {pdf_path.name}: chunk {index + 1}/{attempts} ({label})", flush=True)
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            body = chunk_path.read_text(encoding="utf-8", errors="ignore").strip()
            if body:
                parts.append(f"<!-- chunk {index + 1}/{attempts}: {label} -->\n\n{body}")
                continue
        cmd = [binary, "parse", str(pdf_path)]
        if bounds:
            cmd += ["--pages", f"{bounds[0]}-{bounds[1]}"]
        else:
            cmd += ["--pages", "all"]
        cmd += ["--output", str(chunk_path), "--wait", "3600", *extra_args]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\n[mineru] {pdf_path.name} ({label}): exit code {exc.returncode}")
            continue
        if not chunk_path.exists():
            print(f"\n[mineru] {pdf_path.name} ({label}): no markdown output")
            continue
        body = chunk_path.read_text(encoding="utf-8", errors="ignore").strip()
        if body:
            parts.append(f"<!-- chunk {index + 1}/{attempts}: {label} -->\n\n{body}")
    if len(parts) < attempts:
        print(f"[mineru] {pdf_path.name}: {attempts - len(parts)} of {attempts} chunks missing; will retry later", flush=True)
        return None
    return "\n\n".join(parts)

def write_markdown(target: Path, text: str, keep_image_links: bool) -> None:
    if not keep_image_links:
        text = IMAGE_MD_RE.sub("", text)
    target.write_text(text.strip() + "\n", encoding="utf-8")

def collect_pdfs(pdf_dir: Path) -> list:
    return sorted(
        path
        for path in list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
        if not path.name.startswith("._") and path.name != ".DS_Store"
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF corpus to markdown: markitdown fast path with MinerU OCR fallback for scanned books."
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=("auto", "markitdown", "mineru"), default="auto",
                        help="auto: markitdown first, MinerU handles scanned/failed PDFs")
    parser.add_argument("--min-chars-per-page", type=int, default=50,
                        help="PDFs below this average are treated as scanned and routed to MinerU")
    parser.add_argument("--jobs", type=int, default=4, help="parallel workers for the markitdown pass")
    parser.add_argument("--force", action="store_true", help="re-convert even if markdown already exists")
    parser.add_argument("--chunk-pages", type=int, default=100,
                        help="page-range chunk size for MinerU OCR to stay under its 1h job limit; 0 disables chunking")
    parser.add_argument("--pdfs", default="",
                        help="comma-separated filename globs to convert, e.g. '*弗洛伊德文集1[012]*'")
    parser.add_argument("--keep-image-links", action="store_true", help="keep ![...](...) links instead of stripping them")
    parser.add_argument(
        "--mineru-args",
        default="",
        help='extra MinerU parse flags, e.g. "--tier standard --verbose"',
    )
    args = parser.parse_args()

    pdfs = collect_pdfs(args.pdf_dir)
    patterns = [p.strip() for p in args.pdfs.split(",") if p.strip()]
    if patterns:
        pdfs = [p for p in pdfs if any(fnmatch(p.name, pat) for pat in patterns)]
    if not pdfs:
        print("No PDFs found" + (" matching --pdfs patterns" if patterns else f" in {args.pdf_dir}"))
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = {}
    for pdf in pdfs:
        target = args.out_dir / f"{sanitize_stem(pdf.stem)}.md"
        if target.exists() and not args.force:
            continue
        targets[pdf] = target

    if not targets:
        print("All PDFs already converted. Use --force to re-convert.")
        return

    print(f"{len(targets)} PDFs to convert (mode={args.mode})")
    extra_args = shlex.split(args.mineru_args)
    finished, needs_mineru, failed = [], [], []

    if args.mode in ("auto", "markitdown"):
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(convert_with_markitdown, pdf): pdf for pdf in targets}
            for future in tqdm(as_completed(futures), total=len(futures), desc="markitdown"):
                pdf = futures[future]
                text = future.result()
                if args.mode == "markitdown":
                    if text:
                        write_markdown(targets[pdf], text, args.keep_image_links)
                        finished.append(pdf)
                    else:
                        failed.append(pdf)
                    continue
                pages = count_pages(pdf)
                if text and not looks_scanned(text, pages, args.min_chars_per_page):
                    write_markdown(targets[pdf], text, args.keep_image_links)
                    finished.append(pdf)
                else:
                    needs_mineru.append(pdf)

    if args.mode == "mineru":
        needs_mineru = list(targets)

    if needs_mineru:
        binary = find_mineru_binary()
        if not binary:
            print("mineru/magic-pdf not found on PATH; these PDFs were skipped:")
            failed.extend(needs_mineru)
        else:
            print(f"\n{len(needs_mineru)} PDFs routed to OCR via {binary}")
            tmp_root = args.out_dir / "_mineru_tmp"
            for pdf in tqdm(needs_mineru, desc="mineru"):
                work = tmp_root / sanitize_stem(pdf.stem)
                if args.force:
                    shutil.rmtree(work, ignore_errors=True)
                text = convert_with_mineru(binary, pdf, work, extra_args, args.chunk_pages)
                if text:
                    write_markdown(targets[pdf], text, args.keep_image_links)
                    finished.append(pdf)
                else:
                    failed.append(pdf)
                if text:
                    shutil.rmtree(work, ignore_errors=True)

    print(f"\nDone: {len(finished)} converted, {len(failed)} failed")
    for pdf in failed:
        print(f"  FAILED {pdf.name}")

if __name__ == "__main__":
    main()
