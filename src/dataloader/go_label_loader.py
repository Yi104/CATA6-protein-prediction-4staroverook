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


def build_label_dictionary(df, go2idx):
    """
    Build:
        protein_id → multi-hot vector
    """
    label_dict = {}

    # group by protein
    grouped = df.groupby("entry_id")

    for pid, group in grouped:
        vec = torch.zeros(len(go2idx), dtype=torch.float32)

        for term in group["term"].values:
            idx = go2idx[term]
            vec[idx] = 1.0

        label_dict[pid] = vec

    return label_dict
