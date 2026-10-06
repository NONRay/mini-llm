import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from prepare_psycho_data import clean_markdown
from quality_filter_psycho import cache_key, split_chunks

root = SCRIPT_DIR.parent
parser = argparse.ArgumentParser(description="导出单本心理学书籍的 QC 纯文本")
parser.add_argument(
    "--book",
    required=True,
    help="data/psycho/markdown 下的 Markdown 文件名。",
)
parser.add_argument("--model", default="Qwen3-8B")
parser.add_argument(
    "--audit",
    type=Path,
    default=root / "data/raw/psycho_qc_audit.jsonl",
    help="质量审核记录路径。",
)
parser.add_argument(
    "--clean-only",
    action="store_true",
    help="只应用规则清洗，不读取 Qwen 审核结果。",
)
args = parser.parse_args()
book = args.book

records = {}
for line in args.audit.open(encoding='utf-8'):
    r = json.loads(line)
    if r['source'] == book:
        records[r["key"]] = r

text = clean_markdown((root / 'data/psycho/markdown' / book).read_text(encoding='utf-8', errors='replace'))
chunks = list(split_chunks(text, 1800))

out_dir = root / 'data/raw/psycho_books'
out_dir.mkdir(exist_ok=True)
if args.clean_only:
    output = out_dir / (Path(book).stem + '.clean.txt')
    output.write_text(text + "\n", encoding="utf-8")
    print(f"规则清洗完成｜字符 {len(text)}｜输出 {output}")
else:
    matched = []
    missing = []
    for index, chunk in enumerate(chunks):
        key = cache_key(args.model, book, index, chunk)
        if key not in records:
            missing.append(index)
        else:
            matched.append((chunk, records[key]["decision"]))

    if missing:
        raise RuntimeError(
            f"当前文本有 {len(missing)}/{len(chunks)} 块尚未审核，"
            "不能套用旧版块索引；请先重新运行 quality_filter_psycho.py。"
        )

    kept = [chunk for chunk, decision in matched if decision != "drop"]
    output = out_dir / (Path(book).stem + '.txt')
    output.write_text('\n\n'.join(kept) + "\n", encoding='utf-8')
    print(f'总块 {len(chunks)}｜保留 {len(kept)}｜丢弃 {len(chunks) - len(kept)}｜输出 {output}')
