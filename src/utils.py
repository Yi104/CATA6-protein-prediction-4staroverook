import os
import random
from dataclass import dataclass
from typing import Optional

import torch
import numpy as np

def set_seed(seed: int =42 ) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainingConfig:
    train_terms_path: str = "data/train_terms.tsv"
    embeddings_path: str = "data/embeddings/train_embeddings.pt"
    train_ids_path: Optional[str] = None  # e.g. data/split/train_ids.txt
    val_ids_path: Optional[str] = None  # e.g. data/split/val_ids.txt
    batch_size: int = 256
    num_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 1024
    dropout: float = 0.3
    num_workers: int = 4
    output_dir: str = "models"
    model_name: str = "mlp_baseline.pt"

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)



