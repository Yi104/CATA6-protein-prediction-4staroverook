# src/training/train_mlp.py
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.dataloader.dataset import create_dataloaders
from src.evaluation.fmax import compute_fmax
from src.models.mlp import MLPBaseline
from src.utils import TrainingConfig, ensure_dir, get_device, set_seed


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    bce_loss: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = bce_loss(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Returns: (best_f1, best_threshold)
    """
    model.eval()
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="Val", leave=False):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    y_prob = torch.sigmoid(torch.from_numpy(np.concatenate(all_logits, axis=0))).numpy()
    y_true = np.concatenate(all_targets, axis=0)

    best_f1, best_t = compute_fmax(y_true, y_prob)
    return best_f1, best_t


def run_training(cfg: TrainingConfig) -> None:
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, term2idx, idx2term = create_dataloaders(cfg)

    # Determine embedding dim from first batch
    sample_batch = next(iter(train_loader))
    x_sample, y_sample = sample_batch
    input_dim = x_sample.shape[1]
    output_dim = y_sample.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=output_dim,
        dropout=cfg.dropout,
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    ensure_dir(cfg.output_dir)
    best_f1 = -1.0
    best_model_path = os.path.join(cfg.output_dir, cfg.model_name)
    best_threshold = 0.5

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device, bce_loss)
        print(f"Train loss: {train_loss:.4f}")

        val_f1, val_t = eval_epoch(model, val_loader, device)
        print(f"Val Fmax (simplified): {val_f1:.4f} @ threshold={val_t:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_threshold = val_t
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "term2idx": term2idx,
                    "idx2term": idx2term,
                    "best_threshold": best_threshold,
                    "config": cfg.__dict__,
                },
                best_model_path,
            )
            print(f"New best model saved to {best_model_path}")

    print(f"\nTraining complete. Best Fmax={best_f1:.4f} at threshold={best_threshold:.3f}")


if __name__ == "__main__":
    cfg = TrainingConfig(
        train_terms_path="data/train_terms.tsv",
        embeddings_path="data/embeddings/train_embeddings.pt",
        train_ids_path="data/split/train_ids.txt",  # TODO: create from CD-HIT
        val_ids_path="data/split/val_ids.txt",
        batch_size=256,
        num_epochs=10,
        lr=1e-3,
        hidden_dim=1024,
        dropout=0.3,
        num_workers=4,
        output_dir="models",
        model_name="mlp_baseline.pt",
    )
    run_training(cfg)
