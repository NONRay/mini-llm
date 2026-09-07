import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEM_PROMPT = "你是一个认真、准确、简洁的中文助手。"
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize Chinese instruction data and export plain-text corpus for SFT."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=ROOT / "data/raw/sft_zh_seed.jsonl",
        help="Local JSONL file in instruction format.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset name to load instead of local JSONL.",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Dataset split to load from Hugging Face.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=ROOT / "data/raw/sft_zh_normalized.jsonl",
        help="Path to the normalized JSONL output.",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=ROOT / "data/raw/sft_zh_train.txt",
        help="Path to the plain-text SFT corpus.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt inserted into every sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of normalized samples.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before writing outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def merge_instruction_and_input(instruction: str, extra_input: str) -> str:
    instruction = clean_text(instruction)
    extra_input = clean_text(extra_input)
    if instruction and extra_input:
        return f"{instruction}\n\n补充信息：\n{extra_input}"
    return instruction or extra_input


def format_sample(system_prompt: str, instruction: str, output: str) -> str:
    return (
        f"系统：{system_prompt}\n\n"
        f"用户：{instruction}\n\n"
        f"助手：{output}"
    )


def parse_messages(messages, source: str) -> list[dict]:
    normalized = []
    system_prompt = DEFAULT_SYSTEM_PROMPT
    turns = []

    for item in messages or []:
        role = clean_text(item.get("role") or item.get("from") or "")
        content = clean_text(item.get("content") or item.get("value") or "")
        if not role or not content:
            continue

        role = role.lower()
        if role in {"system"}:
            system_prompt = content
            continue
        if role in {"human", "user"}:
            turns.append(("user", content))
            continue
        if role in {"assistant", "gpt", "bot"}:
            turns.append(("assistant", content))

    history = []
    for role, content in turns:
        if role == "user":
            history.append(("user", content))
            continue

        user_messages = [value for prev_role, value in history if prev_role == "user"]
        if not user_messages:
            continue

        instruction = "\n\n".join(user_messages[-2:])
        sample = {
            "system": system_prompt,
            "instruction": instruction,
            "input": "",
            "output": content,
            "source": source,
        }
        sample["text"] = format_sample(system_prompt, instruction, content)
        normalized.append(sample)
        history.append(("assistant", content))

    return normalized


def normalize_record(record: dict, default_system_prompt: str, source: str) -> list[dict]:
    if "messages" in record:
        return parse_messages(record["messages"], source)
    if "conversations" in record:
        return parse_messages(record["conversations"], source)

    instruction = clean_text(
        record.get("instruction")
        or record.get("prompt")
        or record.get("question")
        or record.get("query")
        or record.get("title")
    )
    extra_input = clean_text(record.get("input") or record.get("context") or "")
    output = clean_text(
        record.get("output")
        or record.get("response")
        or record.get("answer")
        or record.get("completion")
    )
    system_prompt = clean_text(record.get("system") or default_system_prompt)

    instruction = merge_instruction_and_input(instruction, extra_input)
    if not instruction or not output:
        return []

    sample = {
        "system": system_prompt,
        "instruction": instruction,
        "input": extra_input,
        "output": output,
        "source": source,
    }
    sample["text"] = format_sample(system_prompt, instruction, output)
    return [sample]


def iter_local_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_hf_records(dataset_name: str, config_name: str | None, split_name: str) -> Iterable[dict]:
    dataset = load_dataset(dataset_name, config_name, split=split_name)
    for row in dataset:
        yield dict(row)


def collect_samples(args) -> list[dict]:
    if args.hf_dataset:
        records = iter_hf_records(args.hf_dataset, args.hf_config, args.hf_split)
        source_name = args.hf_dataset
    else:
        records = iter_local_records(args.input_jsonl)
        source_name = str(args.input_jsonl.name)

    samples = []
    seen_texts = set()

    for record in records:
        normalized_records = normalize_record(
            record=record,
            default_system_prompt=args.system_prompt,
            source=source_name,
        )
        for sample in normalized_records:
            text = sample["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            samples.append(sample)

            if args.limit is not None and len(samples) >= args.limit:
                return samples

    return samples


def write_outputs(samples: list[dict], output_jsonl: Path, output_text: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_text.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as jsonl_out:
        for sample in samples:
            jsonl_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with output_text.open("w", encoding="utf-8") as text_out:
        for sample in samples:
            text_out.write(sample["text"])
            text_out.write("\n\n")


def main():
    args = parse_args()
    samples = collect_samples(args)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(samples)

    write_outputs(
        samples=samples,
        output_jsonl=args.output_jsonl,
        output_text=args.output_text,
    )

    print(f"samples: {len(samples)}")
    print(f"normalized_jsonl: {args.output_jsonl}")
    print(f"plain_text: {args.output_text}")


if __name__ == "__main__":
    main()
