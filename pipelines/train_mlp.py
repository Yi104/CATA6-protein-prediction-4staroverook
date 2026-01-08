"""
pipelines/train_mlp.py

Train an MLP baseline for CAFA-style GO term prediction using
ESM2 protein-level embeddings (GELU activation only).

Saves:
- checkpoints/go_vocab.json
- checkpoints/best_mlp.pt (selected by Fmax)


# ---------------------------------------------------------------------------
# CHANGE LOG (2026-01-07): Ancestor-Closed GO IC Computation
# ---------------------------------------------------------------------------
# We intentionally compute GO Information Content (IC) on an
# *ancestor-closed* label dictionary, i.e. for each protein we include
# both directly annotated GO terms and all their ancestors in the GO DAG.
#
# WHY THIS IS NECESSARY:
# - GO is a hierarchical ontology (DAG). If a protein is annotated with
#   a child term, it is implicitly annotated with all its parent terms.
# - Computing IC from *direct annotations only* systematically underestimates
#   the frequency of high-level (parent) GO terms.
# - This leads to inflated IC values for generic terms and an overly steep
#   IC distribution, which destabilizes IC-weighted inference.
#
# DESIGN DECISION:
# - Ancestor closure is applied ONLY during IC statistics computation.
# - Ancestor closure is NOT applied during inference or submission.
#
# Additional stabilization:
# - Ontology root terms (MF/BP/CC roots) may be optionally excluded from the
#   ancestor-closed label sets to avoid extreme frequency anchoring.
# This separation is intentional:
#   * IC computation: respects ontology semantics for correct statistics
#   * Inference: avoids ancestor propagation to prevent error amplification
#
# This design aligns with CAFA-style evaluation practice:
# hierarchy-aware metrics + conservative prediction sets.
# ---------------------------------------------------------------------------

"""

from datetime import datetime
import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from goatools.obo_parser import GODag

from src.dataloader.embedding_loader import load_embeddings_h5
from src.dataloader.go_label_loader import (
    load_go_terms,
    build_go_vocabulary,
    build_label_dictionary_sparse,
    build_label_dictionary_set
)
from src.dataloader.dataset import ProteinDataset
from src.models.mlp import MLPClassifier
from src.training.trainer import Trainer


from src.ontology.go_ic import compute_go_ic
from src.ontology.go_ontology import load_go_ontology, build_ontology_index
from src.ontology.propagation import propagate_ancestors

from src.evaluation.metrics import compute_fmax_hierarchical


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMB_PATH = "data/embeddings/esm2_650M_trainval_concat_2560.h5"
TERMS_PATH = "data/raw/Train/train_terms.tsv"
TRAIN_ID_PATH = "data/processed/train_id_40.csv"
VAL_ID_PATH = "data/processed/val_id_40.csv"
OBO_PATH = "data/raw/Train/go-basic.obo"

CHECKPOINT_DIR = "checkpoints"
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best_mlp.pt")
GO_VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "go_vocab.json")



BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5
NUM_WORKERS = 0
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_id_list(path: str) -> set[str]:
    df = pd.read_csv(path)
    return set(df.iloc[:, 0].astype(str))


def normalize_vocab_to_idx2go(vocab):
    # normalize to idx2go list (same logic you used)
    if isinstance(vocab, list):
        return vocab
    if isinstance(vocab, dict) and all(k.isdigit() for k in vocab.keys()):
        return [vocab[str(i)] for i in range(len(vocab))]
    if isinstance(vocab, dict) and all(isinstance(k, str) and k.startswith("GO:") for k in vocab.keys()):
        max_i = max(vocab.values())
        idx2go = [None] * (max_i + 1)
        for go_id, i in vocab.items():
            idx2go[i] = go_id
        if any(x is None for x in idx2go):
            raise ValueError("go_vocab.json (go2idx) is missing indices")
        return idx2go
    raise ValueError("Unrecognized GO vocab format")




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
    print("\nStep1: Loading embeddings...")
    emb_dict, emb_info = load_embeddings_h5(EMB_PATH, return_info=True)
    input_dim = emb_info["dimensionality"]
    print(f"Loaded {len(emb_dict)} embeddings (dim={input_dim})")

    # --------------------------------------------------
    # 2) Load GO terms + build vocabulary
    # --------------------------------------------------
    print("\nStep 2 : Loading GO terms...")
    df_terms = load_go_terms(TERMS_PATH)
    go2idx, idx2go = build_go_vocabulary(df_terms)
    label_dict = build_label_dictionary_sparse(df_terms, go2idx)

    # Sanity check: idx2go must not contain None
    if any(x is None for x in idx2go):
        bad = [i for i, x in enumerate(idx2go) if x is None][:10]
        raise ValueError(f"idx2go contains None at indices: {bad}")

    output_dim = len(go2idx)
    print(f"GO terms: {output_dim}")
    print(f"Proteins with labels: {len(label_dict)}")

    # Save GO vocab (idx -> GO)
    idx2go_json = {str(i): term for i, term in enumerate(idx2go)}
    if not os.path.exists(GO_VOCAB_PATH):
        with open(GO_VOCAB_PATH, "w") as f:
            json.dump(idx2go_json, f)
        print(f"Saved GO vocabulary → {GO_VOCAB_PATH}")
    else:
        print(f"GO vocabulary already exists → {GO_VOCAB_PATH}")

    # -------------------------------
    # 3) Load GO DAG (ontology semantics)
    # -------------------------------
    print("\nStep 3: Loading GO DAG...")
    godag = GODag(OBO_PATH)
    print("idx2go size:", len(idx2go))
    print("godag terms:", len(godag))

    # --------------------------------------------------
    # 4) Compute GO IC (for CAFA-style weighted metric)
    # --------------------------------------------------

    IC_PATH = "checkpoints/go_ic.pt"

    if not os.path.exists(IC_PATH):
        print("\n Step 4: Computing GO IC weights...")
        label_go_dict = build_label_dictionary_set(df_terms)

        label_go_dict_closed ={}
        for pid,go_set in label_go_dict.items():
            label_go_dict_closed[pid] = propagate_ancestors(go_set,godag,drop_roots = False)

        go_ic = compute_go_ic(label_go_dict_closed)
        # Save with metadata to avoid future confusion (backward compatible)
        torch.save(
            {
                "go_ic": go_ic,
                "ancestor_closed": True,
                "source": os.path.basename(TERMS_PATH),
                "ic_definition": "IC(go)=-log(freq(go)) computed on ancestor-closed train labels",
            },
            IC_PATH,
        )
        print(f"Saved GO IC to {IC_PATH}")
    else:
        payload = torch.load(IC_PATH, map_location="cpu")
        go_ic = payload["go_ic"] if isinstance(payload, dict) and "go_ic" in payload else payload
        print(f"Loaded GO IC from {IC_PATH} (keys={len(go_ic)})")

    # Quick sanity check for IC distribution
    roots = ["GO:0003674", "GO:0008150", "GO:0005575"]
    print("IC roots:", {r: go_ic.get(r, None) for r in roots})

    vals = list(go_ic.values())
    print(
        f"IC stats — min: {min(vals):.4f}, "
        f"median: {np.median(vals):.4f}, "
        f"max: {max(vals):.4f}"
    )

    # --------------------------------------------------
    # 5) Load train / val splits
    # --------------------------------------------------
    train_ids = load_id_list(TRAIN_ID_PATH)
    val_ids = load_id_list(VAL_ID_PATH)

    # --------------------------------------------------
    # 6) Build datasets
    # --------------------------------------------------
    print("\nStep 6: Building datasets...")
    train_ds = ProteinDataset(emb_dict, label_dict, train_ids, output_dim = len(go2idx))
    val_ds = ProteinDataset(emb_dict, label_dict, val_ids, output_dim = len(go2idx))

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # --------------------------------------------------
    # 7) Model + optimizer + loss
    # --------------------------------------------------
    print("\nStep 7: Initializing model...")
    model = MLPClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()



    # --------------------------------------------------
    # 8) Ontology split indices (MF/BP/CC)
    # --------------------------------------------------
    go2ont = load_go_ontology(OBO_PATH)
    mf_idx, bp_idx, cc_idx = build_ontology_index(idx2go, go2ont)
    print(f"\n Step 8 Ontology split — MF: {len(mf_idx)}, BP: {len(bp_idx)}, CC: {len(cc_idx)}")


    # --------------------------------------------------
    # 9) Training loop (checkpoint by Fmax)
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        idx2go=idx2go,
        godag=godag,
        ic_path=IC_PATH, # ic_weighted, else ic_path = None for unweighted.
    )

    print("\nStarting training...\n")
    best_score = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = trainer.train_one_epoch(train_loader)

        # ALL terms (optional to display)
        val_loss, f_all, t_all = trainer.validate_with_metrics(
            val_loader,
            thresholds = None,
            max_proteins = None,
            top_k = 200,
            term_idx = None,
            )

        # Full validation: MF/BP/CC
        _, f_mf, t_mf = trainer.validate_with_metrics(val_loader, term_idx=mf_idx, top_k=100)
        _, f_cc, t_cc = trainer.validate_with_metrics(val_loader, term_idx=cc_idx, top_k=100)
        _, f_bp, t_bp = trainer.validate_with_metrics(val_loader, term_idx=bp_idx, top_k=300)

        f_avg = (f_mf + f_cc + f_bp) / 3.0

        print(
            f"\n{ts}"
            f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"ALL {f_all:.4f}@{t_all:.2f} | "
            f"MF {f_mf:.4f}@{t_mf:.2f} | CC {f_cc:.4f}@{t_cc:.2f} | BP {f_bp:.4f}@{t_bp:.2f} | "
            f"AVG {f_avg:.4f}"
        )

        LOG_PATH = os.path.join(CHECKPOINT_DIR, "trian.log")
        log_f = open(LOG_PATH, "w")

        line = (
            f"\n{ts}"
            f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"ALL {f_all:.4f}@{t_all:.2f} | "
            f"MF {f_mf:.4f}@{t_mf:.2f} | CC {f_cc:.4f}@{t_cc:.2f} | BP {f_bp:.4f}@{t_bp:.2f} | "
            f"AVG {f_avg:.4f}\n"
        )
        print(line, end="")
        log_f.write(line)
        log_f.flush()

        # --------------------------------------------------
        # Save per-epoch checkpoint (for ensembling / debugging)
        # --------------------------------------------------
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"mlp_{run_ts}_epoch{epoch:02d}.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "f_avg": f_avg,
                "f_mf": f_mf, "t_mf": t_mf,
                "f_cc": f_cc, "t_cc": t_cc,
                "f_bp": f_bp, "t_bp": t_bp,
                "input_dim": input_dim,
                "output_dim": output_dim,
            },
            epoch_ckpt_path,
        )
        print(f"  Saved epoch checkpoint → {epoch_ckpt_path}")

        # Choose a single checkpoint selection rule:
        #  use f_avg (macro avg across ontologies)
        score_for_ckpt = f_avg

        if score_for_ckpt > best_score:
            best_score = score_for_ckpt
            print("  New best score — saving checkpoint")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "f_all": f_all,
                    "t_all": t_all,
                    "f_mf": f_mf, "t_mf": t_mf,
                    "f_cc": f_cc, "t_cc": t_cc,
                    "f_bp": f_bp, "t_bp": t_bp,
                    "f_avg": f_avg,
                    # keep old keys for compatibility
                    "fmax": f_all,
                    "threshold": t_all,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                },
                BEST_CKPT_PATH,
            )

    print("\nTraining complete.")
    print(f"Best macro AVG Fmax: {best_score:.4f}")
    print(f"Checkpoint saved to: {BEST_CKPT_PATH}")

    log_f.close()





if __name__ == "__main__":
    main()
