"""
Defines ProteinDataset — a PyTorch Dataset for protein-level classification.
Each item returned is a tuple:
    (embedding_vector, label_vector)

Embedding vectors come from ESM2 protein-level embeddings (e.g., 2560-dim).
Label vectors come from GO-term annotations (multi-hot).

Example:
    x = tensor([2560 dims])
    y = tensor([num_GO_terms dims], dtype=float32)

This Dataset operates entirely on preloaded dictionaries to ensure:
- fast random access
- reproducibility
- flexible ID-based splits (train/val/test)
"""


import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein classification tasks (e.g., GO prediction).

    Parameters
    ----------
    emb_dict : dict[str, torch.Tensor]
        Mapping from protein_id → embedding tensor of shape (D,).
        Example: emb_dict["A0A0C5B5G6"] → tensor([2560])

    label_dict : dict[str, torch.Tensor]
        Mapping from protein_id → label vector of shape (num_labels,).
        Example: label_dict["A0A0C5B5G6"] → tensor([0,1,0,...])

    protein_ids : list[str]
        Ordered list of protein IDs to include in this dataset.
        This defines the training / validation split explicitly.

    Notes
    -----
    - The Dataset does *not* load embeddings from disk. Use
      embedding_loader.load_embeddings_h5() before passing to this Dataset.
    - The Dataset ensures that only IDs present in BOTH dictionaries are used.
    - Does not assume fixed embedding dimension — works with 1280, 2560, etc.
    """

    def __init__(self, emb_dict, label_dict, protein_ids):
        self.emb = emb_dict
        self.labels = label_dict

        # Filter IDs to only those available in both dicts
        self.ids = [
            pid for pid in protein_ids
            if pid in emb_dict and pid in label_dict
        ]

        if len(self.ids) == 0:
            raise ValueError("No overlapping protein IDs between embeddings and labels.")

        # Validate embedding dimensions (debug safety)
        example_pid = self.ids[0]
        self.embedding_dim = self.emb[example_pid].shape[-1]
        self.num_labels = self.labels[example_pid].shape[-1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Returns
        -------
        embedding : torch.Tensor  shape = (embedding_dim,)
        labels    : torch.Tensor  shape = (num_labels,)
        """
        pid = self.ids[index]
        x = self.emb[pid]
        y = self.labels[pid]
        return x, y
