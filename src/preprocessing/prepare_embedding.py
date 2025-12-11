"""
prepare_embedding.py

This module performs the preprocessing needed for ESM2 embedding:

1. Load train/val protein IDs from CSV.
2. Load UniProt FASTA into dictionary {uniprot_id: sequence}.
3. Match ID lists to FASTA sequences -> (id, seq) pairs.
4. Create length-based bins for efficient embedding
   (short, mid, long, ultra-long).

These functions are shared by:
    - esm_mean.py
    - esm_concat.py
    - downstream MLP/GNN tasks
"""

import os
from src.dataloader.sequence_loader import parse_uniprot_fasta
from src.dataloader.id_loader import load_id_list


# ------------------------------------------------------------
#       BUILD SEQUENCE LISTS: train, val, test
# ------------------------------------------------------------
def build_sequence_pairs(id_set, fasta_dict):
    """
    Given a set of protein IDs and a FASTA dict:
    return only those (id, sequence) pairs that exist in FASTA.
    """
    return [(pid, fasta_dict[pid]) for pid in id_set if pid in fasta_dict]


def load_train_val_test_sequences(base_dir):
    """
    Load raw FASTA + train/val ID CSVs + test FASTA.

    Returns:
        train_seqs: list[(pid, seq)]
        val_seqs:   list[(pid, seq)]
        test_seqs:  list[(pid, seq)]
    """
    RAW_TRAIN_FASTA = f"{base_dir}/data/raw/Train/train_sequences.fasta"
    RAW_TEST_FASTA  = f"{base_dir}/data/raw/Test/testsuperset.fasta"
    TRAIN_ID_PATH   = f"{base_dir}/data/processed/train_id_40.csv"
    VAL_ID_PATH     = f"{base_dir}/data/processed/val_id_40.csv"

    # Load ID lists
    train_ids = load_id_list(TRAIN_ID_PATH)
    val_ids   = load_id_list(VAL_ID_PATH)

    # Load FASTA dictionaries
    train_fasta = parse_uniprot_fasta(RAW_TRAIN_FASTA)
    test_fasta  = parse_uniprot_fasta(RAW_TEST_FASTA)

    # Build aligned (id, sequence) pairs
    train_seqs = build_sequence_pairs(train_ids, train_fasta)
    val_seqs   = build_sequence_pairs(val_ids, train_fasta)
    test_seqs  = list(test_fasta.items())  # already (id, seq)

    return train_seqs, val_seqs, test_seqs


# ------------------------------------------------------------
#       LENGTH BINNING FOR GPU/MEMORY EFFICIENCY
# ------------------------------------------------------------
def classify_by_length(seq_list):
    """
    Classify sequences by length for batching:
        short  <= 1022
        mid    1023–2048
        long   2049–5000
        ultra  > 5000
    """
    bins = {
        "short_<=1022": [],
        "mid_1023_2048": [],
        "long_2049_5000": [],
        "ultra_>5000": [],
    }
    for pid, seq in seq_list:
        L = len(seq)
        if L <= 1022:
            bins["short_<=1022"].append((pid, seq))
        elif L <= 2048:
            bins["mid_1023_2048"].append((pid, seq))
        elif L <= 5000:
            bins["long_2049_5000"].append((pid, seq))
        else:
            bins["ultra_>5000"].append((pid, seq))
    return bins


def load_length_bins(train_seqs, val_seqs, test_seqs):
    """
    Given train/val/test sequence lists,
    return (train_bins, val_bins, test_bins)
    each of which is a dict of length categories.
    """
    return (
        classify_by_length(train_seqs),
        classify_by_length(val_seqs),
        classify_by_length(test_seqs),
    )
