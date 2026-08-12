import numpy as np
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
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y
