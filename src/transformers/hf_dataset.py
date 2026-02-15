from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class TextDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[int], ids: List[int]):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["id"] = torch.tensor(self.ids[idx], dtype=torch.long)
        return item
