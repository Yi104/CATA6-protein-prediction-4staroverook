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
from src.ontology.go_ontology import load_go_ontology, build_ontology_index

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
NUM_WORKERS = 0

# Replace global TOP_K
TOP_K_MF = 100 # change to 100 later
TOP_K_CC = 100 # change to 100 later
TOP_K_BP = 300 # chagne to 300 later

MIN_SCORE_MF = 0.02
MIN_SCORE_CC = 0.02
MIN_SCORE_BP = 0.01



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
    # 3) Load GO vocabulary and ontology
    # --------------------------------------------------
    with open(GO_VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    # ---------- normalize vocab to idx2go: List[str] -------------
    if isinstance(vocab,list):
        idx2go = vocab

    elif isinstance(vocab,dict):
        # case: {"0": "GO:....", "1": "GO:....", ...}
        if all(isinstance(k,str) and k.isdigit() for k in vocab.keys()):
            idx2go = [vocab[str(i)] for i in range(len(vocab))]
        # case: {"GO:....": 0, "GO:....": 1, ...}
        elif all(isinstance(k, str) and k.startswith("GO:") for k in vocab.keys()):
            max_i = max(vocab.values())
            idx2go = [None] * (max_i + 1)
            for go_id, i in vocab.items():
                idx2go[i] = go_id
            if any(x is None for x in idx2go):
                raise ValueError("go_vocab.json (go2idx) is missing indices")

        else:
            raise ValueError(f"Unrecognized go_vocab.json format: {list(vocab.items())[:3]}")

    else:
        raise ValueError(f"Unrecognized go_vocab.json type: {type(vocab)}")

    print(f"Loaded idx2go with {len(idx2go)} terms. Example: {idx2go[:5]}")


    # ================================
    # 4) GO Ontology (MF / BP / CC)
    # ================================

    OBO_PATH = "data/raw/Train/go-basic.obo"

    print("Loading GO ontology...")
    go2ont = load_go_ontology(OBO_PATH)
    print(f"GO terms with ontology info: {len(go2ont)}")
    mf_idx, bp_idx, cc_idx = build_ontology_index(idx2go, go2ont)

    print(
        f"Ontology split — "
        f"MF: {len(mf_idx)}, "
        f"BP: {len(bp_idx)}, "
        f"CC: {len(cc_idx)}"
    )

    # --------------------------------------------------
    # 5) Build model
    # --------------------------------------------------
    model = MLPClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --------------------------------------------------
    # 6) Inference
    # --------------------------------------------------

    print("\nRunning ontology-aware inference...")
    total_written = 0

    with open(OUTPUT_PATH, "w") as out_f:
        with torch.no_grad():
            for pids, x in tqdm(test_loader, desc="Inference"):
                x = x.to(device, non_blocking=True)

                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()

                for pid, scores in zip(pids, probs):

                    # ---------- MF ----------
                    mf_scores = scores[mf_idx]
                    if mf_scores.size > 0:
                        k = min(TOP_K_MF, mf_scores.size)
                        top_local = np.argpartition(-mf_scores, k - 1)[:k]
                        top_local = top_local[np.argsort(mf_scores[top_local])[::-1]]

                        for j in top_local:
                            score = mf_scores[j]
                            if score < MIN_SCORE_MF:
                                break
                            go_id = idx2go[mf_idx[j]]
                            out_f.write(f"{pid}\t{go_id}\t{score:.4f}\n")
                            total_written += 1

                    # ---------- CC ----------
                    cc_scores = scores[cc_idx]
                    if cc_scores.size > 0:
                        k = min(TOP_K_CC, cc_scores.size)
                        top_local = np.argpartition(-cc_scores, k - 1)[:k]
                        top_local = top_local[np.argsort(cc_scores[top_local])[::-1]]

                        for j in top_local:
                            score = cc_scores[j]
                            if score < MIN_SCORE_CC:
                                break
                            go_id = idx2go[cc_idx[j]]
                            out_f.write(f"{pid}\t{go_id}\t{score:.4f}\n")
                            total_written += 1

                    # ---------- BP ----------
                    bp_scores = scores[bp_idx]
                    if bp_scores.size > 0:
                        k = min(TOP_K_BP, bp_scores.size)
                        top_local = np.argpartition(-bp_scores, k - 1)[:k]
                        top_local = top_local[np.argsort(bp_scores[top_local])[::-1]]

                        for j in top_local:
                            score = bp_scores[j]
                            if score < MIN_SCORE_BP:
                                break
                            go_id = idx2go[bp_idx[j]]
                            out_f.write(f"{pid}\t{go_id}\t{score:.4f}\n")
                            total_written += 1

    print("\nInference complete.")
    print(f"Submission written to: {OUTPUT_PATH}")
    print(f"Total predictions written: {total_written}")


if __name__ == "__main__":
    main()
