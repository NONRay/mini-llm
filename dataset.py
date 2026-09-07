import json

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """从预处理好的 uint32 memmap 二进制中按滑窗切 (x, y) 训练样本。

    每个样本长度为 seq_len + 1：x 取前 seq_len 个 token，
    y 为错位 1 的后 seq_len 个 token（标准的 next-token 目标）。
    """

    def __init__(self, path, seq_len):
        """加载 memmap，避免将完整语料一次性读入内存。"""
        self.data = np.memmap(
            path,
            dtype=np.uint32,
            mode="r",
        )
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        """返回错位一位的输入与 next-token 监督标签。"""
        chunk = np.asarray(
            self.data[idx : idx + self.seq_len + 1],
            dtype=np.int64,
        )
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


class SFTDataset(Dataset):
    """将标准化 SFT JSONL 编码成仅对助手回答计算 loss 的样本。"""

    def __init__(self, path, tokenizer_path, seq_len):
        self.seq_len = seq_len
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.eos_id = self.tokenizer.eos_id()
        self.pad_id = self.eos_id if self.tokenizer.pad_id() < 0 else self.tokenizer.pad_id()
        self.samples = []

        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample = self._build_sample(row)
                if sample is not None:
                    self.samples.append(sample)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, out_type=int)

    def _build_sample(self, row: dict):
        system = str(row.get("system", "")).strip()
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()

        if not instruction or not output:
            return None

        prompt_text = (
            f"系统：{system}\n\n"
            f"用户：{instruction}\n\n"
            f"助手："
        )
        answer_text = output

        prompt_ids = self._encode(prompt_text)
        answer_ids = self._encode(answer_text)
        if self.eos_id >= 0:
            answer_ids = answer_ids + [self.eos_id]

        if not answer_ids:
            return None

        max_prompt_len = max(0, self.seq_len - len(answer_ids))
        if max_prompt_len == 0:
            answer_ids = answer_ids[: self.seq_len]
            if len(answer_ids) < 2:
                return None
            prompt_ids = []
        elif len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        tokens = prompt_ids + answer_ids
        if len(tokens) < 2:
            return None

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        prompt_target_len = max(0, len(prompt_ids) - 1)
        if prompt_target_len > 0:
            y[:prompt_target_len] = -100
        if len(prompt_ids) > 0:
            y[len(prompt_ids) - 1] = answer_ids[0]

        if torch.all(y.eq(-100)):
            return None

        return x, y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        xs, ys = zip(*batch)
        max_len = max(item.size(0) for item in xs)

        batch_x = torch.full((len(xs), max_len), self.pad_id, dtype=torch.long)
        batch_y = torch.full((len(ys), max_len), -100, dtype=torch.long)

        for i, (x, y) in enumerate(zip(xs, ys)):
            batch_x[i, : x.size(0)] = x
            batch_y[i, : y.size(0)] = y

        return batch_x, batch_y
