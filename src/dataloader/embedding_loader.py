"""
embedding_loader.py

Flexible embedding loader for protein-level or residue-level embeddings
stored in HDF5 format.

Supports:
- mean pooling (1280-d)
- mean + max concat (2560-d)
- attention-weighted pooling (future)
- any arbitrary embedding dimensionality
- residue-level representations (L, 1280)

This loader auto-detects embedding type by reading dataset shapes.

Returned structure:
-------------------
{
    "A0A12345": {
        "embedding": tensor([...]),
        "shape": (2560,),
        "level": "protein",     # or "residue"
    },
    ...
}

This allows downstream models to adapt behavior depending on embedding type.
"""

import h5py
import torch
import numpy as np


def load_embeddings_h5(h5_path, return_info=False):
    """
    Load an HDF5 embedding file into memory.

    Parameters
    ----------
    h5_path : str
        Path to the embedding .h5 file.

    return_info : bool
        If True, also return metadata dictionary describing the loaded embeddings.

    Returns
    -------
    emb_dict : dict[str, torch.Tensor]
        Mapping from protein_id → embedding tensor.

    emb_info : dict  (optional)
        Returns:
        {
            'n_proteins': 82404,
            'example_id': 'A0A0C5B5G6',
            'example_shape': (2560,),
            'embedding_level': 'protein',
            'dimensionality': 2560,
    }


        Useful for debugging or downstream tasks.
    """

    emb_dict = {}
    shapes = {}

    with h5py.File(h5_path, "r") as f:
        for pid in f.keys():
            arr = np.array(f[pid])

            # Detect whether protein-level or residue-level
            if arr.ndim == 1:
                level = "protein"
                dim = arr.shape[0] # residue level (L, 1280) if embedding in 1280
            elif arr.ndim == 2:
                level = "residue"
                dim = arr.shape[1]  # embedding dimension (1280)
            else:
                raise ValueError(f"Unexpected embedding shape {arr.shape} for {pid}")

            shapes[pid] = arr.shape
            emb_dict[pid] = torch.tensor(arr, dtype=torch.float32)

    # Compute metadata summary
    if return_info:
        example_pid = next(iter(shapes))
        example_shape = shapes[example_pid]
        info = {
            "n_proteins": len(emb_dict),
            "example_id": example_pid,
            "example_shape": example_shape,
            "embedding_level": "protein" if len(example_shape) == 1 else "residue",
            "dimensionality": example_shape[-1],
            "shapes": shapes,
        }
        return emb_dict, info

    return emb_dict
