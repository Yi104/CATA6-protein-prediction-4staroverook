"""
pipelines/validation_full.py

Full validation for a trained model:
- hierarchical Fmax on ALL terms
- hierarchical Fmax on MF / CC / BP
- macro-average across MF/CC/BP

Callable from notebook:
    from pipelines.validattion_full import run_full_validation

Runnable as script:
    python pipelines/validation_full.py
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from goatools.obo_parser import GODag

from src.dataloader.embedding_loader import load_embeddings_h5
from src.dataloader.go_label_loader import (
    load_go_terms, build_go_vocabulary, build_label_dictionary_sparse, build_label_dictionary_set
)
from src.dataloader.dataset import ProteinDataset
from src.models.mlp import MLPClassifier

from src.evaluation.metrics import compute_fmax_hierarchical
from src.ontology.go_ontology import load_go_ontology, build_ontology_index


# -------------------------
# Configs
# -------------------------

EMB_PATH    = "data/embeddings/esm2_650M_trainval_concat_2560.h5"
TERMS_PATH  = "data/raw/Train/train_terms.tsv"
VAL_ID_PATH = "data/processed/val_id_40.csv"
OBO_PATH    = "data/raw/Train/go-basic.obo"

CKPT_PATH   = "checkpoints/best_mlp.pt"
IC_PATH     = "checkpoints/go_ic.pt"

BATCH_SIZE  = 64
NUM_WORKERS = 0


def load_id_list(path: str) -> set[str]:
    df = pd.read_csv(path)
    return set(df.iloc[:, 0].astype(str))


def run_full_validation(
    model,
    val_loader,
    idx2go,
    godag,
    mf_idx,
    bp_idx,
    cc_idx,
    go_ic=None,
    thresholds=None,
    max_proteins=None,
    topk_all=200,
    topk_mf=100,
    topk_cc=100,
    topk_bp=300,
    device=None,
):
    """
    Library interface: compute full validation metrics from already-loaded objects.
    Returns a dict (easy to log/compare in a notebook).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # ALL
    f_all, t_all = compute_fmax_hierarchical(
        logits, targets, idx2go, godag,
        go_ic=go_ic, thresholds=thresholds,
        max_proteins=max_proteins, topk=topk_all
    )

    # MF
    logits_mf, targets_mf = logits[:, mf_idx], targets[:, mf_idx]
    idx2go_mf = [idx2go[i] for i in mf_idx]
    f_mf, t_mf = compute_fmax_hierarchical(
        logits_mf, targets_mf, idx2go_mf, godag,
        go_ic=go_ic, thresholds=thresholds,
        max_proteins=max_proteins, topk=topk_mf
    )

    # CC
    logits_cc, targets_cc = logits[:, cc_idx], targets[:, cc_idx]
    idx2go_cc = [idx2go[i] for i in cc_idx]
    f_cc, t_cc = compute_fmax_hierarchical(
        logits_cc, targets_cc, idx2go_cc, godag,
        go_ic=go_ic, thresholds=thresholds,
        max_proteins=max_proteins, topk=topk_cc
    )

    # BP
    logits_bp, targets_bp = logits[:, bp_idx], targets[:, bp_idx]
    idx2go_bp = [idx2go[i] for i in bp_idx]
    f_bp, t_bp = compute_fmax_hierarchical(
        logits_bp, targets_bp, idx2go_bp, godag,
        go_ic=go_ic, thresholds=thresholds,
        max_proteins=max_proteins, topk=topk_bp
    )

    f_avg = (float(f_mf) + float(f_cc) + float(f_bp)) / 3.0

    return {
        "f_all": float(f_all), "t_all": float(t_all),
        "f_mf":  float(f_mf),  "t_mf":  float(t_mf),
        "f_cc":  float(f_cc),  "t_cc":  float(t_cc),
        "f_bp":  float(f_bp),  "t_bp":  float(t_bp),
        "f_avg": float(f_avg),
        "n_proteins": int(targets.shape[0]),
        "n_terms_all": int(targets.shape[1]),
        "n_terms_mf": int(len(mf_idx)),
        "n_terms_cc": int(len(cc_idx)),
        "n_terms_bp": int(len(bp_idx)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load embeddings
    emb_dict, emb_info = load_embeddings_h5(EMB_PATH, return_info=True)
    input_dim = emb_info["dimensionality"]
    print(f"Embeddings: {len(emb_dict)} | dim={input_dim}")

    # Build labels/vocab (must match training)
    df_terms = load_go_terms(TERMS_PATH)
    go2idx, idx2go = build_go_vocabulary(df_terms)
    label_dict = build_label_dictionary_sparse(df_terms, go2idx)
    print(f"GO terms: {len(idx2go)} | labeled proteins: {len(label_dict)}")

    # Val dataset/loader
    val_ids = load_id_list(VAL_ID_PATH)
    val_ds = ProteinDataset(emb_dict, label_dict, val_ids, output_dim = len(go2idx))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Val proteins: {len(val_ds)}")

    # Ontology
    godag = GODag(OBO_PATH)
    go2ont = load_go_ontology(OBO_PATH)
    mf_idx, bp_idx, cc_idx = build_ontology_index(idx2go, go2ont)
    print(f"Ontology split — MF: {len(mf_idx)} | BP: {len(bp_idx)} | CC: {len(cc_idx)}")

    # Load checkpoint/model
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = MLPClassifier(input_dim=input_dim, output_dim=ckpt["output_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')} | output_dim={ckpt['output_dim']}")

    # Optional IC
    go_ic = torch.load(IC_PATH) if os.path.exists(IC_PATH) else None
    print("IC loaded:", go_ic is not None)

    # Compute metrics
    results = run_full_validation(
        model=model,
        val_loader=val_loader,
        idx2go=idx2go,
        godag=godag,
        mf_idx=mf_idx,
        bp_idx=bp_idx,
        cc_idx=cc_idx,
        go_ic=go_ic,
        thresholds=None,
        max_proteins=None,
        topk_all=200,
        topk_mf=100,
        topk_cc=100,
        topk_bp=300,
        device=device,
    )

    print(
        f"ALL {results['f_all']:.4f}@{results['t_all']:.2f} | "
        f"MF {results['f_mf']:.4f}@{results['t_mf']:.2f} | "
        f"CC {results['f_cc']:.4f}@{results['t_cc']:.2f} | "
        f"BP {results['f_bp']:.4f}@{results['t_bp']:.2f} | "
        f"AVG {results['f_avg']:.4f}"
    )


if __name__ == "__main__":
    main()
