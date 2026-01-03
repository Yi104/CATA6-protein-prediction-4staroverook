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
    with open(GO_VOCAB_PATH, "w") as f:
        json.dump(idx2go_json, f)
    print(f"Saved GO vocabulary → {GO_VOCAB_PATH}")

    # --------------------------------------------------
    # 2.5) Compute GO IC (for CAFA-style weighted metric)
    # --------------------------------------------------

    IC_PATH = "checkpoints/go_ic.pt"

    if not os.path.exists(IC_PATH):
        print("Computing GO IC weights...")
        label_go_dict = build_label_dictionary_set(df_terms)
        go_ic = compute_go_ic(label_go_dict)
        torch.save(go_ic, IC_PATH)
        print(f"Saved GO IC to {IC_PATH}")
    else:
        print(f"GO IC already exists at {IC_PATH}")

    # --------------------------------------------------
    # 3) Load train / val splits
    # --------------------------------------------------
    train_ids = load_id_list(TRAIN_ID_PATH)
    val_ids = load_id_list(VAL_ID_PATH)

    # --------------------------------------------------
    # 4) Build datasets
    # --------------------------------------------------
    print("\nBuilding datasets...")
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
    # 5) Model + optimizer + loss
    # --------------------------------------------------
    print("\nInitializing model...")
    model = MLPClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()

    # -------------------------------
    # 6) GO vocabulary + GO DAG (for validation metric only)
    # -------------------------------
    with open(GO_VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    idx2go = normalize_vocab_to_idx2go(vocab)

    godag = GODag(OBO_PATH)
    print("idx2go size:", len(idx2go))
    print("godag terms:", len(godag))

    # --------------------------------------------------
    # 7) Ontology split indices (MF/BP/CC)
    # --------------------------------------------------
    go2ont = load_go_ontology(OBO_PATH)
    mf_idx, bp_idx, cc_idx = build_ontology_index(idx2go, go2ont)
    print(f"Ontology split — MF: {len(mf_idx)}, BP: {len(bp_idx)}, CC: {len(cc_idx)}")




    # --------------------------------------------------
    # 8) Training loop (checkpoint by Fmax)
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        idx2go=idx2go,
        godag=godag,
        ic_path=IC_PATH,
    )

    print("\nStarting training...\n")
    best_fmax = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = trainer.train_one_epoch(train_loader)

        # ALL terms (optional to display)
        val_loss, f_all, t_all = trainer.validate_with_metrics(
            val_loader,
            thresholds = None,
            max_proteins = None,
            topk = 200,
            term_idx = None,
            )

        # Full validation: MF/BP/CC
        _, f_mf, t_mf = trainer.validate_with_metrics(val_loader, term_idx=mf_idx, topk=100)
        _, f_cc, t_cc = trainer.validate_with_metrics(val_loader, term_idx=cc_idx, topk=100)
        _, f_bp, t_bp = trainer.validate_with_metrics(val_loader, term_idx=bp_idx, topk=300)

        f_avg = (f_mf + f_cc + f_bp) / 3.0

        print(
            f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"ALL {f_all:.4f}@{t_all:.2f} | "
            f"MF {f_mf:.4f}@{t_mf:.2f} | CC {f_cc:.4f}@{t_cc:.2f} | BP {f_bp:.4f}@{t_bp:.2f} | "
            f"AVG {f_avg:.4f}"
        )

        # Choose a single checkpoint selection rule:
        # Option A (recommended): use f_avg (macro avg across ontologies)
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

    # for epoch in range(1, EPOCHS + 1):
    #     train_loss = trainer.train_one_epoch(train_loader)
    #     val_loss, fmax, best_t = trainer.validate_with_metrics(val_loader)
    #
    #     print(
    #         f"Epoch {epoch:02d} | "
    #         f"Train Loss: {train_loss:.4f} | "
    #         f"Val Loss: {val_loss:.4f} | "
    #         f"Fmax: {fmax:.4f} @ t={best_t:.2f}"
    #     )
    #
    #     if fmax > best_fmax:
    #         best_fmax = fmax
    #         print("  New best Fmax — saving checkpoint")
    #
    #         torch.save(
    #             {
    #                 "model_state_dict": model.state_dict(),
    #                 "epoch": epoch,
    #                 "fmax": fmax,
    #                 "threshold": best_t,
    #                 "input_dim": input_dim,
    #                 "output_dim": output_dim,
    #             },
    #             BEST_CKPT_PATH,
    #         )




if __name__ == "__main__":
    main()
