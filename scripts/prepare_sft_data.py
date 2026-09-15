import argparse
import ast
import json
import random
import re
from collections import defaultdict
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
        "--hf-configs",
        type=str,
        default=None,
        help="Optional comma-separated Hugging Face dataset config names.",
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
    parser.add_argument(
        "--sample-strategy",
        type=str,
        choices=["sequential", "coig_cqia_balanced"],
        default="sequential",
        help="Sampling strategy used before writing outputs.",
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


def parse_literal(value):
    if isinstance(value, (dict, list, tuple)):
        return value
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def normalize_string_list(value) -> list[str]:
    value = parse_literal(value)
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = clean_text(value)
        return [cleaned] if cleaned else []
    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in value:
            cleaned = clean_text(item)
            if cleaned:
                normalized.append(cleaned)
        return normalized
    return [clean_text(value)]


def normalize_task_type(task_type) -> dict[str, list[str]]:
    parsed = parse_literal(task_type)
    if isinstance(parsed, dict):
        major = normalize_string_list(parsed.get("major"))
        minor = normalize_string_list(parsed.get("minor"))
        return {"major": major, "minor": minor}
    return {"major": [], "minor": []}


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return bool(value)


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


def enrich_sample_metadata(sample: dict, record: dict, source: str, source_config: str | None) -> dict:
    task_type = normalize_task_type(record.get("task_type"))
    sample["task_type"] = task_type
    sample["domain"] = normalize_string_list(record.get("domain"))
    sample["answer_from"] = clean_text(record.get("answer_from") or "")
    sample["human_verified"] = parse_bool(record.get("human_verified"))
    sample["copyright"] = clean_text(record.get("copyright") or "")
    sample["source"] = source
    sample["source_config"] = source_config or ""
    return sample


def normalize_record(
    record: dict,
    default_system_prompt: str,
    source: str,
    source_config: str | None = None,
) -> list[dict]:
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
    }
    sample["text"] = format_sample(system_prompt, instruction, output)
    return [enrich_sample_metadata(sample, record, source, source_config)]


def iter_local_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_hf_configs(config_name: str | None, config_names: str | None) -> list[str | None]:
    if config_names:
        configs = [item.strip() for item in config_names.split(",") if item.strip()]
        return configs
    return [config_name]


def iter_hf_records(
    dataset_name: str,
    config_name: str | None,
    config_names: str | None,
    split_name: str,
) -> Iterable[tuple[dict, str | None]]:
    configs = parse_hf_configs(config_name, config_names)
    for cfg in configs:
        dataset = load_dataset(dataset_name, cfg, split=split_name)
        for row in dataset:
            yield dict(row), cfg


def select_coig_cqia_balanced(samples: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or len(samples) <= limit:
        return samples

    rng = random.Random(seed)
    subset_buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for sample in samples:
        subset = sample.get("source_config") or "default"
        minors = sample.get("task_type", {}).get("minor") or ["未分类"]
        bucket_key = " | ".join(minors)
        subset_buckets[subset][bucket_key].append(sample)

    subset_names = list(subset_buckets.keys())
    for subset_name in subset_names:
        for bucket_name in subset_buckets[subset_name]:
            rng.shuffle(subset_buckets[subset_name][bucket_name])

    subset_round_robin = {
        subset_name: sorted(subset_buckets[subset_name].keys())
        for subset_name in subset_names
    }
    subset_bucket_index = {subset_name: 0 for subset_name in subset_names}
    selected = []
    seen_texts = set()

    while len(selected) < limit:
        progressed = False
        for subset_name in subset_names:
            bucket_names = subset_round_robin[subset_name]
            if not bucket_names:
                continue

            for _ in range(len(bucket_names)):
                idx = subset_bucket_index[subset_name] % len(bucket_names)
                bucket_name = bucket_names[idx]
                subset_bucket_index[subset_name] = idx + 1
                bucket = subset_buckets[subset_name][bucket_name]
                if not bucket:
                    continue

                sample = bucket.pop()
                text = sample["text"]
                if text in seen_texts:
                    continue

                seen_texts.add(text)
                selected.append(sample)
                progressed = True
                break

            if len(selected) >= limit:
                break

        if not progressed:
            break

    return selected


def collect_samples(args) -> list[dict]:
    if args.hf_dataset:
        records = iter_hf_records(
            args.hf_dataset,
            args.hf_config,
            args.hf_configs,
            args.hf_split,
        )
        source_name = args.hf_dataset
    else:
        records = ((record, None) for record in iter_local_records(args.input_jsonl))
        source_name = str(args.input_jsonl.name)

    samples = []
    seen_texts = set()

    for record, source_config in records:
        normalized_records = normalize_record(
            record=record,
            default_system_prompt=args.system_prompt,
            source=source_name,
            source_config=source_config,
        )
        for sample in normalized_records:
            text = sample["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            samples.append(sample)

            if args.limit is not None and args.sample_strategy == "sequential" and len(samples) >= args.limit:
                return samples

    if args.sample_strategy == "coig_cqia_balanced":
        return select_coig_cqia_balanced(samples, args.limit, args.seed)

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
