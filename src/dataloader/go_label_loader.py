"""
go_label_loader.py

Loads CAFA6 train_terms.tsv and produces:
1. A GO vocabulary (go2idx and idx2go)
2. A dictionary mapping:
       protein_id → multi-hot label vector (torch.float32)

train_terms.tsv has columns:
- entry_id : UniProt protein accession
- term     : GO term (GO:XXXXXXX)
- aspect   : F/P/C (ontology branch)
"""

import pandas as pd
import torch
from collections import defaultdict


def load_go_terms(term_file):
    """
    Read CAFA6 term file (train_terms.tsv) and return:
        df : pandas DataFrame
    - Pandas reads columns as C1, C2, C3
    - The first row contains the true header:
        EntryID, term, aspect
    """
    df = pd.read_csv(term_file, sep="\t")
    if list(df.columns) == ["C1", "C2", "C3"]:
        df.columns = df.iloc[0] # promote first row to header
        df = df.iloc[1:].reset_index(drop = True)

    df.columns = [c.strip() for c in df.columns] # remove extra space
    df= df.rename(columns ={"EntryID": "entry_id"})
    required = {"entry_id", "term", "aspect"}
    if not required.issubset(set(df.columns)):
        raise ValueError("Missing required columns: {}".format(required))

    return df


def build_go_vocabulary(df):
    """
    Build GO → index mapping from the dataframe.
    Returns:
        go2idx : dict
        idx2go : list
    """
    unique_terms = sorted(df["term"].unique())
    go2idx = {term: i for i, term in enumerate(unique_terms)}
    idx2go = unique_terms
    return go2idx, idx2go


# def build_label_dictionary(df, go2idx):
#     """
#     Build:
#         protein_id -> multi-hot vector
#         This is dense index (later oom, change to sparse index (2026/01/03) see function below
#     """
#     label_dict = {}
#
#     # group by protein
#     grouped = df.groupby("entry_id")
#
#     for pid, group in grouped:
#         vec = torch.zeros(len(go2idx), dtype=torch.float32) # this step cause OOM
#
#
#         for term in group["term"].values:
#             idx = go2idx[term]
#             vec[idx] = 1.0
#
#         label_dict[pid] = vec
#
#     return label_dict

def build_label_dictionary_sparse(df, go2idx):
    """
    Build:
    protein_idx -> list[int] (go term indexes)
    used for training/validation dataset to construct dense multi-hot
    Why sparse:
    - avoids storing one dense vector per protein (huge RAM)
    - dense vector is only needed per sample/batch
    """
    label_idx = defaultdict(list)
    for pid, term in zip(df["entry_id"].astype(str), df["term"].astype(str)):
        if term in go2idx:
            label_idx[pid].append(go2idx[term])

        # optional: remove duplicates to be safe (rare but cheap)
    for pid in label_idx:
        label_idx[pid] = list(set(label_idx[pid]))

    return dict(label_idx)

def build_label_dictionary_set(df):
    """
    Build:
        protein_id -> set[str] (GO IDs)
    Used for IC computation and any GO-ID-based statistics.
    no propogate_ancestor (term) --> flat
    """
    label_go = defaultdict(set)

    for pid, term in zip(df["entry_id"].astype(str), df["term"].astype(str)):
        label_go[pid].add(term)

    return dict(label_go)