
import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein classification tasks (e.g., GO prediction).

    IMPORTANT DESIGN NOTE (20260103 YJ)
    ---------------------
    We deliberately store labels in *sparse form* (list of GO indices)
    instead of precomputing dense multi-hot vectors for all proteins.

    Reason:
    - num_GO_terms can be very large (~20k–30k)
    - pre-allocating one dense vector per protein leads to extreme memory usage
    - dense label vectors are only needed *per batch*, not globally

    At __getitem__ time, sparse indices are converted into a dense multi-hot vector
    of length output_dim.

    This design:
    - keeps memory usage low and stable
    - follows PyTorch best practices
    - avoids OOM / IDE crashes


    IMPLEMENTATION NOTE (2026-01-04 YJ)
    --------------------------------
    When densifying sparse labels, we use vectorized index_fill_ instead of
    Python-level loops (e.g., `y[idx] = 1`).

    Rationale:
    - avoids millions of Python assignments per epoch
    - leverages PyTorch's optimized backend
    - scales better for large output spaces (e.g., CAFA / GO prediction)

     Notes
    -----
    - The Dataset does *not* load embeddings from disk. Use
      embedding_loader.load_embeddings_h5() before passing to this Dataset.
    - The Dataset ensures that only IDs present in BOTH dictionaries are used.
    - Does not assume fixed embedding dimension — works with 1280, 2560, etc.
    """

    def __init__(self, emb_dict, label_dict, protein_ids, output_dim):
        """
                Parameters
                ----------
                emb_dict : dict[str, torch.Tensor]
                    protein_id → embedding tensor, shape (D,)

                label_idx_dict : dict[str, list[int]]
                    protein_id → list of GO-term indices (sparse labels)
                    Example: {"P12345": [12, 87, 2031]}

                protein_ids : list[str]
                    IDs defining the split (train / val)

                output_dim : int
                    Total number of GO terms (size of label space)
                """
        self.emb = emb_dict
        self.labels = label_dict
        self.output_dim = output_dim

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


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Returns
        -------
        embedding : torch.Tensor  shape = (embedding_dim,)
        labels    : torch.Tensor  Dense multi-hot vector of shape (output_dim,)
            Constructed *on-the-fly* from sparse indices.
        """
        pid = self.ids[index]
        x = self.emb[pid]
        y = torch.zeros(self.output_dim, dtype=torch.float32)

        # Sparse → dense (vectorized)
        idxs = self.labels[pid]
        if isinstance(idxs, torch.Tensor):
            inds = idxs.long()
        else:
            inds = torch.tensor(list(idxs), dtype=torch.long)

        if inds.numel() > 0:
            # Safety: clip invalid indices (defensive programming)
            inds = inds[(inds >= 0) & (inds < self.output_dim)]
            y.index_fill_(0, inds, 1.0)

        return x, y
