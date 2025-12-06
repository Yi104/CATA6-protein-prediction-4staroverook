from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.utils import TrainingConfig

class ProteinDataset(Dataset):
    """
    returns (embedding, multihot_labels) for each protein in the dataset
    """
    def __init__(
            self,
            embeddings: Dict[str, torch.Tensor],
            labels: Dict[str, torch.Tensor],
            protein_ids: Optional[List[int]] = None,
    ) -> None:
        self.embeddings = embeddings
        self.labels = labels

        all_ids = sorted(list(labels.keys()))
        if protein_ids is None:
            self.protein_ids = all_ids
        else:
            # Filter to only those present in labels + embeddings
            self.protein_ids = [
                pid for pid in protein_ids
                if pid in labels and pid in embeddings
            ]

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, index: int):
        pid = self.protein_ids[index]
        emb = self.embeddings[pid]
        y = self.labels[pid]
        return emb, y
