"""
pipelines/train_mlp.py

Train an MLP baseline for CAFA-style GO term prediction using
ESM2 protein-level embeddings (GELU activation only).

Saves:
- checkpoints/go_vocab.json
- checkpoints/best_mlp.pt (selected by Fmax)
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataloader.embedding_loader import load_embeddings_h5
from src.dataloader.go_label_loader import (
    load_go_terms,
    build_go_vocabulary,
    build_label_dictionary,
)
from src.dataloader.dataset import ProteinDataset
from src.models.mlp import MLPClassifier
from src.training.trainer import Trainer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMB_PATH = "data/embeddings/esm2_650M_trainval_concat_2560.h5"
TERMS_PATH = "data/raw/Train/train_terms.tsv"
TRAIN_ID_PATH = "data/processed/train_id_40.csv"
VAL_ID_PATH = "data/processed/val_id_40.csv"

CHECKPOINT_DIR = "checkpoints"
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best_mlp.pt")
GO_VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "go_vocab.json")

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5
NUM_WORKERS = 0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_id_list(path: str) -> set[str]:
    df = pd.read_csv(path)
    return set(df.iloc[:, 0].astype(str))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # 1) Load embeddings
    # --------------------------------------------------
    print("\nLoading embeddings...")
    emb_dict, emb_info = load_embeddings_h5(EMB_PATH, return_info=True)
    input_dim = emb_info["dimensionality"]
    print(f"Loaded {len(emb_dict)} embeddings (dim={input_dim})")

    # --------------------------------------------------
    # 2) Load GO terms + build vocabulary
    # --------------------------------------------------
    print("\nLoading GO terms...")
    df_terms = load_go_terms(TERMS_PATH)
    go2idx, idx2go = build_go_vocabulary(df_terms)
    label_dict = build_label_dictionary(df_terms, go2idx)

    output_dim = len(go2idx)
    print(f"GO terms: {output_dim}")
    print(f"Proteins with labels: {len(label_dict)}")

    # Save GO vocab (idx -> GO)
    idx2go_json = {str(i): term for i, term in enumerate(idx2go)}
    with open(GO_VOCAB_PATH, "w") as f:
        json.dump(idx2go_json, f)
    print(f"Saved GO vocabulary → {GO_VOCAB_PATH}")

    # --------------------------------------------------
    # 3) Load train / val splits
    # --------------------------------------------------
    train_ids = load_id_list(TRAIN_ID_PATH)
    val_ids = load_id_list(VAL_ID_PATH)

    # --------------------------------------------------
    # 4) Build datasets
    # --------------------------------------------------
    print("\nBuilding datasets...")
    train_ds = ProteinDataset(emb_dict, label_dict, train_ids)
    val_ds = ProteinDataset(emb_dict, label_dict, val_ids)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # --------------------------------------------------
    # 5) Model + optimizer + loss
    # --------------------------------------------------
    print("\nInitializing model...")
    model = MLPClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
    )

    # --------------------------------------------------
    # 6) Training loop (checkpoint by Fmax)
    # --------------------------------------------------
    print("\nStarting training...\n")
    best_fmax = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss, fmax, best_t = trainer.validate_with_metrics(val_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Fmax: {fmax:.4f} @ t={best_t:.2f}"
        )

        if fmax > best_fmax:
            best_fmax = fmax
            print("  New best Fmax — saving checkpoint")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "fmax": fmax,
                    "threshold": best_t,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                },
                BEST_CKPT_PATH,
            )

    print("\nTraining complete.")
    print(f"Best Fmax: {best_fmax:.4f}")
    print(f"Checkpoint saved to: {BEST_CKPT_PATH}")


if __name__ == "__main__":
    main()
