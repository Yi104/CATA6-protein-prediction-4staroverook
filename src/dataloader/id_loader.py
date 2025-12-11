"""
id_loader.py

Utility for loading protein IDs (train/val splits) from CSV files.
"""

import pandas as pd


def load_id_list(csv_path):
    """
    Load a CSV file with a column 'protein_id' and return a set of IDs.

    This matches the CAFA6 train/val ID split files you tested in Colab.

    Returns
    -------
    set[str]
        Set of UniProt IDs.
    """
    df = pd.read_csv(csv_path)
    return set(df["protein_id"].astype(str).tolist())
