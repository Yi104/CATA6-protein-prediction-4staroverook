"""
esm_concat.py

This script generates **protein-level embeddings** using ESM2 (650M) with
**mean pooling + max pooling concatenation** to produce a 2560-dimensional vector.

It uses:
- FASTA files from data/raw/
- train_id_40.csv and val_id_40.csv from data/processed/
- length-based bins for memory-efficient embedding
- HDF5 output for fast lookups during training

This script is intended for **full embedding extraction on GPU (4090) or comparable device**.

not recommended for colab A100 if not using pro+
"""

import os
import torch
import h5py
from tqdm import tqdm
import esm

from src.preprocessing.prepare_embedding import (
    load_train_val_test_sequences,
    load_length_bins
)


# -------------------------------------------------------------------------
#                      CORE CONCAT POOLING FUNCTION
# -------------------------------------------------------------------------
def embed_concat_batch(model, batch_converter, device, batch):
    """
    Embed a batch of proteins using ESM2 and return a dictionary:
        {protein_id: embedding (2560-dim numpy array)}

    Steps:
        1. Tokenize sequences with batch_converter
        2. Forward pass through ESM2 layer 33
        3. Extract residue embeddings (L x 1280)
        4. Compute mean pooling (1280)
        5. Compute max pooling (1280)
        6. Concatenate → (2560,)
    """
    labels, strs, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)

    reps = out["representations"][33].cpu()   # shape: [B, L+2, 1280]

    embeddings = {}
    for j, (pid, seq) in enumerate(batch):
        L = len(seq)
        # Remove BOS/EOS → keep residues only
        emb = reps[j, 1:L+1, :]                # [L, 1280]

        # Protein-level pooling
        mean_emb = emb.mean(dim=0)             # [1280]
        max_emb  = emb.max(dim=0).values       # [1280]
        protein_emb = torch.cat([mean_emb, max_emb], dim=0)  # [2560]

        embeddings[pid] = protein_emb.numpy()

    return embeddings


# -------------------------------------------------------------------------
#            BIN-WISE EMBEDDING (SAVES MEMORY & IMPROVES SPEED)
# -------------------------------------------------------------------------
def embed_bin_concat(model, batch_converter, device, seqs, h5_file,
                     batch_size=8, truncate_to=3500):
    """
    Embed a bin of sequences (same length range) and save each result to HDF5.

    This function:
        - skips already-embedded proteins (resume support)
        - applies optional truncation for very long sequences (>3500 AA)
        - processes in batches for GPU efficiency
    """
    done = set(h5_file.keys())
    seqs = [(pid, s) for pid, s in seqs if pid not in done]
    print(f"  Remaining in this bin: {len(seqs)}")

    for i in tqdm(range(0, len(seqs), batch_size)):
        batch = seqs[i:i + batch_size]

        # Optional truncation for ultra-long proteins
        if truncate_to:
            batch = [(pid, s[:truncate_to]) for pid, s in batch]

        batch_emb = embed_concat_batch(model, batch_converter, device, batch)

        # Save to HDF5
        for pid, emb in batch_emb.items():
            h5_file.create_dataset(pid, data=emb)


# -------------------------------------------------------------------------
#                          MAIN EMBEDDING DRIVER
# -------------------------------------------------------------------------
def run_concat_embedding(base_dir, output_h5_path):
    """
    Full pipeline:
        1. Load ESM2 model
        2. Load train/val/test sequences
        3. Create length bins
        4. Embed each bin (short → ultra-long)
        5. Save protein-level embeddings to HDF5
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    # Load model
    print("Loading ESM2-650M model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # Load sequence data
    print("Loading sequences...")
    train_seqs, val_seqs, test_seqs = load_train_val_test_sequences(base_dir)
    train_bins, val_bins, test_bins = load_length_bins(train_seqs, val_seqs, test_seqs)

    print(f"Train: {len(train_seqs)}  Val: {len(val_seqs)}  Test: {len(test_seqs)}")

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    # HDF5 output file
    with h5py.File(output_h5_path, "a") as h5_file:
        print("Embedding training set by bins...")
        for bin_name, seqs in train_bins.items():
            print(f"Processing bin: {bin_name}")
            embed_bin_concat(model, batch_converter, device, seqs, h5_file)

    print("\n=== CONCAT EMBEDDING COMPLETE ===")
    print("Saved to:", output_h5_path)


# -------------------------------------------------------------------------
#                              ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    BASE = os.path.expanduser("~/CAFA6")
    OUTPUT = f"{BASE}/data/embeddings/esm2_650M_train_concat_2560.h5"

    run_concat_embedding(BASE, OUTPUT)
