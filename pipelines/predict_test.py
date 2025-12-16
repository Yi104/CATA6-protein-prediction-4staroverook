"""
pipelines/predict_test.py

Run inference on test proteins and generate Kaggle submission
using a trained GELU-MLP model.
"""

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from src.dataloader.embedding_loader import load_embeddings_h5
from src.models.mlp import MLPClassifier

import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_EMB_PATH = "data/embeddings/esm2_650M_test_concat_2560.h5"
CHECKPOINT_PATH = "checkpoints/best_mlp.pt"
GO_VOCAB_PATH = "checkpoints/go_vocab.json"

OUTPUT_DIR = "submissions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "submission.tsv")

BATCH_SIZE = 32
TOP_K = 500
NUM_WORKERS = 0


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class TestDataset(Dataset):
    def __init__(self, emb_dict):
        self.ids = list(emb_dict.keys())
        self.emb = emb_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        return pid, self.emb[pid]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # 1) Load test embeddings
    # --------------------------------------------------
    print("\nLoading test embeddings...")
    emb_dict, emb_info = load_embeddings_h5(TEST_EMB_PATH, return_info=True)
    input_dim = emb_info["dimensionality"]
    print(f"Loaded {len(emb_dict)} test proteins")

    test_ds = TestDataset(emb_dict)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------
    # 2) Load checkpoint
    # --------------------------------------------------
    print("\nLoading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    threshold = ckpt["threshold"]
    output_dim = ckpt["output_dim"]

    print(f"Threshold: {threshold:.3f}")
    print(f"Output dim: {output_dim}")

    # --------------------------------------------------
    # 3) Load GO vocabulary
    # --------------------------------------------------
    with open(GO_VOCAB_PATH, "r") as f:
        idx2go = json.load(f)

    # --------------------------------------------------
    # 4) Build model
    # --------------------------------------------------
    model = MLPClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --------------------------------------------------
    # 5) Inference
    # --------------------------------------------------

    print("\nRunning inference...")
    total_written = 0

    with open(OUTPUT_PATH, "w") as out_f:
        with torch.no_grad():
            for pids, x in tqdm(test_loader, desc="Inference"):
                x = x.to(device, non_blocking=True)

                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()

                for pid, scores in zip(pids, probs):
                    # Partial top-K (O(N))
                    k = min(TOP_K, scores.shape[0])
                    topk_idx = np.argpartition(-scores, k - 1)[:k]

                    # Sort only top-K
                    topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

                    for idx in topk_idx:
                        score = scores[idx]
                        if score <= 0.0:
                            continue

                        score_fmt = float(f"{score:.3g}")
                        go_id = idx2go[str(idx)]

                        out_f.write(f"{pid}\t{go_id}\t{score_fmt}\n")
                        total_written += 1

    print("\nInference complete.")
    print(f"Submission written to: {OUTPUT_PATH}")
    print(f"Total predictions written: {total_written}")


if __name__ == "__main__":
    main()
