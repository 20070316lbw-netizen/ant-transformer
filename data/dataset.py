"""
dataset.py — SST-2 情感分类数据集

使用 HuggingFace datasets + transformers tokenizer。

SST-2 标签：
  0 = negative
  1 = positive
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from config import AntConfig


class SST2Dataset(Dataset):

    def __init__(self, split: str, config: AntConfig):
        """
        Args:
            split:  'train' | 'validation'
            config: AntConfig
        """
        # 从 HuggingFace Hub 加载 SST-2（GLUE 子集）
        self.data = load_dataset("glue", "sst2", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.max_seq_len = config.max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        enc = self.tokenizer(
            item["sentence"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # [T]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [T]
            "label":          torch.tensor(item["label"], dtype=torch.long),
        }


def get_dataloaders(config: AntConfig) -> tuple[DataLoader, DataLoader]:
    """返回 (train_loader, val_loader)"""
    train_ds = SST2Dataset("train",      config)
    val_ds   = SST2Dataset("validation", config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader
