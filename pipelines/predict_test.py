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
from goatools.obo_parser import GODag
from src.ontology.propagation import propagate_ancestors
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_EMB_PATH = "data/embeddings/esm2_650M_test_concat_2560.h5"
CHECKPOINT_PATH = "checkpoints/best_mlp.pt"
GO_VOCAB_PATH = "checkpoints/go_vocab.json"


BATCH_SIZE = 32
NUM_WORKERS = 0

# Replace global TOP_K
TOP_K_MF = 100 # change to 100 later
TOP_K_CC = 100 # change to 100 later
TOP_K_BP = 300 # chagne to 300 later

# Per-ontology thresholds from IC-weighted full validation (Fmax-optimal)
THRESH_MF = 0.02 #0.17
THRESH_CC = 0.02# 0.11
THRESH_BP = 0.01# 0.06

# Use thresholds as the score cutoff in inference
MIN_SCORE_MF = THRESH_MF
MIN_SCORE_CC = THRESH_CC
MIN_SCORE_BP = THRESH_BP


OUTPUT_DIR = "submissions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

run_tag = f"mf{MIN_SCORE_MF:.2f}_cc{MIN_SCORE_CC:.2f}_bp{MIN_SCORE_BP:.2f}_" + \
          datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"submission_{run_tag}.tsv")


meta = {
    "checkpoint": CHECKPOINT_PATH,
    "topk": {"MF": TOP_K_MF, "CC": TOP_K_CC, "BP": TOP_K_BP},
    "threshold": {"MF": MIN_SCORE_MF, "CC": MIN_SCORE_CC, "BP": MIN_SCORE_BP},
    "propagate_ancestors": True,
}
with open(OUTPUT_PATH.replace(".tsv", ".json"), "w") as f:
    json.dump(meta, f, indent=2)


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

def propagate_with_max_scores(go_score: dict, godag):
    """
    Add all ancestors of predicted GO terms.

    Scoring rule (professional default):
    - ancestor score = max score among its predicted descendants

    Why:
    - preserves score monotonicity up the DAG
    - avoids artificially inflating confidence
    - aligns inference output with hierarchy-aware evaluation
    """
    out = dict(go_score)

    for go_id, s in list(go_score.items()):
        if go_id not in godag:
            continue
        for parent in godag[go_id].get_all_parents():
            prev = out.get(parent)
            if prev is None or s > prev:
                out[parent] = float(s)

    return out

def soft_hierarchical_consistency_reweighting(go_score: dict, godag, gamma=0.7, require_parent=False):
    """
    Soft hierarchy consistency without adding ancestors.
    - If a term's parent is not predicted, down-weight the term.
    - Optionally: require parent presence to keep the term (hard filter).

    gamma: 0<gamma<=1, smaller => stronger penalty (e.g. 0.6~0.85)
    require_parent: if True, drop term if none of its direct parents are predicted

    20260105 YJ - new method to proceed ancestor propogation
    Hierarchy-aware score reweighting.
    Instead of propagating predictions to ancestor terms, we apply a parent-aware score reweighting strategy that
    softly penalizes predicted child terms whose direct parents are absent from the prediction set.
    This approach encourages hierarchical consistency while avoiding the error amplification commonly observed in full ancestor propagation.
    """
    out = dict(go_score)
    predicted = set(go_score.keys())

    for go_id, s in list(go_score.items()):
        if go_id not in godag:
            continue

        # direct parents (not all ancestors)
        parents = set(getattr(godag[go_id], "parents", []))  # goatools term has .parents
        if not parents:
            continue

        has_parent = any(p.id in predicted for p in parents)  # parent nodes are GO objects
        if not has_parent:
            if require_parent:
                out.pop(go_id, None)
            else:
                out[go_id] = float(s) * gamma

    return out


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
    print("\nStep 1: Loading test embeddings...")
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
    print("\nStep 2: Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    threshold_all = ckpt.get("threshold", None)
    print(f"Checkpoint ALL-threshold: {threshold_all:.3f}" if threshold_all is not None else "No threshold in ckpt")

    output_dim = ckpt["output_dim"]
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

    print(f"Step 3: Loaded idx2go with {len(idx2go)} terms. Example: {idx2go[:5]}")


    # ================================
    # 4) GO Ontology (MF / BP / CC)
    # ================================

    OBO_PATH = "data/raw/Train/go-basic.obo"

    print("\nStep 4: Loading GO ontology...")
    go2ont = load_go_ontology(OBO_PATH)
    print(f"GO terms with ontology info: {len(go2ont)}")
    mf_idx, bp_idx, cc_idx = build_ontology_index(idx2go, go2ont)
    godag = GODag(OBO_PATH) # add DAG for evaluation and post processing

    print(
        f"\nStep 4 Ontology split — "
        f"MF: {len(mf_idx)}, "
        f"BP: {len(bp_idx)}, "
        f"CC: {len(cc_idx)}"
    )

    # --------------------------------------------------
    # 5) Build model
    # --------------------------------------------------
    print("\nStep 5: Building model ...")
    model = MLPClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --------------------------------------------------
    # 6) Inference
    # --------------------------------------------------

    print("\nStep 6: Running ontology-aware inference...")
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

                        mf_pred = {}
                        for j in top_local:
                            score = mf_scores[j]
                            if score < MIN_SCORE_MF:
                                break # continue?
                            go_id = idx2go[mf_idx[j]]

                            # keep best score if duplicate occurs:
                            prev = mf_pred.get(go_id)
                            if prev is None or score > prev:
                                mf_pred[go_id] = score

                        # Hierarchy completion (add ancestors prediction)
                        # mf_pred = propagate_with_max_scores(mf_pred, godag)  don't do hierachy  Jan 6th
                        # mf_pred = soft_hierarchical_consistency_reweighting(mf_pred, godag, gamma=0.7, require_parent=False)

                        for go_id, score in mf_pred.items():
                            out_f.write(f"{pid}\t{go_id}\t{score:.4f}\n")
                            total_written += 1


                    # ---------- CC ----------
                    cc_scores = scores[cc_idx]
                    if cc_scores.size > 0:
                        k = min(TOP_K_CC, cc_scores.size)
                        top_local = np.argpartition(-cc_scores, k - 1)[:k]
                        top_local = top_local[np.argsort(cc_scores[top_local])[::-1]]

                        cc_pred = {}
                        for j in top_local:
                            score = float(cc_scores[j])
                            if score < MIN_SCORE_CC:
                                break
                            go_id = idx2go[cc_idx[j]]
                            prev = cc_pred.get(go_id)
                            if prev is None or score > prev:
                                cc_pred[go_id] = score
                        # Hierarchy completion (add ancestors prediction)
                        # cc_pred = propagate_with_max_scores(cc_pred, godag)
                        # cc_pred = soft_hierarchical_consistency_reweighting(cc_pred, godag, gamma=0.7, require_parent=False)

                        for go_id, score in cc_pred.items():
                            out_f.write(f"{pid}\t{go_id}\t{score:.4f}\n")
                            total_written += 1

                    # ---------- BP ----------
                    bp_scores = scores[bp_idx]
                    if bp_scores.size > 0:
                        k = min(TOP_K_BP, bp_scores.size)
                        top_local = np.argpartition(-bp_scores, k - 1)[:k]
                        top_local = top_local[np.argsort(bp_scores[top_local])[::-1]]

                        bp_pred = {}
                        for j in top_local:
                            score = float(bp_scores[j])
                            if score < MIN_SCORE_BP:
                                break
                            go_id = idx2go[bp_idx[j]]
                            prev = bp_pred.get(go_id)
                            if prev is None or score > prev:
                                bp_pred[go_id] = score

                        # Hierarchy completion (add ancestors prediction)
                        # bp_pred = propagate_with_max_scores(bp_pred, godag)
                        # cc_pred = soft_hierarchical_consistency_reweighting(cc_pred, godag, gamma=0.7, require_parent=False)

                        for go_id, score in bp_pred.items():
                            out_f.write(f"{pid}\t{go_id}\t{score:.4f}\n")
                            total_written += 1

    print("\nInference complete.")
    print(f"Submission written to: {OUTPUT_PATH}")
    print(f"Total predictions written: {total_written}")


if __name__ == "__main__":
    main()
