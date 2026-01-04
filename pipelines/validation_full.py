"""
pipelines/validation_full.py

Full validation for a trained model:
- hierarchical Fmax on ALL terms
- hierarchical Fmax on MF / CC / BP
- macro-average across MF/CC/BP

Callable from notebook:
    from pipelines.validation_full import main

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
    load_go_terms,
    build_go_vocabulary,
    build_label_dictionary_sparse,
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


def collect_logits_targets(model, val_loader, device):
    """
    Forward once over val_loader and cache logits/targets on CPU.
    This avoids running the model twice when comparing unweighted vs IC-weighted.
    """
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return logits, targets


def run_full_validation_from_tensors(
    logits,
    targets,
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
):
    """
    Compute full validation metrics from pre-collected logits/targets tensors.
    Returns a dict.
    """

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
        "ic_weighted": bool(go_ic is not None),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load embeddings
    emb_dict, emb_info = load_embeddings_h5(EMB_PATH, return_info=True)
    input_dim = emb_info["dimensionality"]
    print(f"Embeddings: {len(emb_dict)} | dim={input_dim}")

    # 2) Build labels/vocab (must match training)
    df_terms = load_go_terms(TERMS_PATH)
    go2idx, idx2go = build_go_vocabulary(df_terms)
    label_idx_dict = build_label_dictionary_sparse(df_terms, go2idx)
    print(f"GO terms: {len(idx2go)} | labeled proteins: {len(label_idx_dict)}")

    # 3) Val dataset/loader
    val_ids = load_id_list(VAL_ID_PATH)
    val_ds = ProteinDataset(
        emb_dict=emb_dict,
        label_idx_dict=label_idx_dict,
        protein_ids=val_ids,
        output_dim=len(go2idx),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    print(f"Val proteins: {len(val_ds)}")

    # 4) Ontology
    godag = GODag(OBO_PATH)
    go2ont = load_go_ontology(OBO_PATH)
    mf_idx, bp_idx, cc_idx = build_ontology_index(idx2go, go2ont)
    print(f"Ontology split — MF: {len(mf_idx)} | BP: {len(bp_idx)} | CC: {len(cc_idx)}")

    # 5) Load checkpoint/model
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = MLPClassifier(input_dim=input_dim, output_dim=ckpt["output_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')} | output_dim={ckpt['output_dim']}")

    # 6) Load IC (optional)
    go_ic = torch.load(IC_PATH) if os.path.exists(IC_PATH) else None
    print("IC loaded:", go_ic is not None)

    # 7) Collect logits/targets once
    logits, targets = collect_logits_targets(model, val_loader, device)
    print(f"Cached logits/targets: logits={tuple(logits.shape)} targets={tuple(targets.shape)}")

    # 8) Compute UNWEIGHTED and WEIGHTED on same tensors
    results_unw = run_full_validation_from_tensors(
        logits, targets, idx2go, godag, mf_idx, bp_idx, cc_idx,
        go_ic=None,
        thresholds=None,
        max_proteins=None,
        topk_all=200,
        topk_mf=100,
        topk_cc=100,
        topk_bp=300,
    )

    results_w = run_full_validation_from_tensors(
        logits, targets, idx2go, godag, mf_idx, bp_idx, cc_idx,
        go_ic=go_ic,
        thresholds=None,
        max_proteins=None,
        topk_all=200,
        topk_mf=100,
        topk_cc=100,
        topk_bp=300,
    )

    print("\n--- Unweighted ---")
    print(
        f"ALL {results_unw['f_all']:.4f}@{results_unw['t_all']:.2f} | "
        f"MF {results_unw['f_mf']:.4f}@{results_unw['t_mf']:.2f} | "
        f"CC {results_unw['f_cc']:.4f}@{results_unw['t_cc']:.2f} | "
        f"BP {results_unw['f_bp']:.4f}@{results_unw['t_bp']:.2f} | "
        f"AVG {results_unw['f_avg']:.4f}"
    )

    if go_ic is None:
        print("\n--- IC-weighted ---")
        print("IC not found; skipping weighted metrics.")
        return

    print("\n--- IC-weighted ---")
    print(
        f"ALL {results_w['f_all']:.4f}@{results_w['t_all']:.2f} | "
        f"MF {results_w['f_mf']:.4f}@{results_w['t_mf']:.2f} | "
        f"CC {results_w['f_cc']:.4f}@{results_w['t_cc']:.2f} | "
        f"BP {results_w['f_bp']:.4f}@{results_w['t_bp']:.2f} | "
        f"AVG {results_w['f_avg']:.4f}"
    )

    # Optional: assert monotonic expectation
    # It's common (not guaranteed) that weighted <= unweighted.
    # If weighted > unweighted substantially, re-check IC logic or go_ic coverage.
    if results_w["f_all"] > results_unw["f_all"] + 1e-6:
        print("\n[WARN] IC-weighted ALL is higher than unweighted ALL. "
              "This can happen but is uncommon; double-check IC logic and go_ic coverage.")


if __name__ == "__main__":
    main()
