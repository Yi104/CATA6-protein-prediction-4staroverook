from __future__ import annotations

import numpy as np
from typing import Tuple


def precision_recall_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    """
    Simplified micro-averaged precision/recall/F1 for multi-label.
    y_true, y_prob: (N, C) arrays
    """
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def compute_fmax(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_thresholds: int = 50,
) -> Tuple[float, float]:
    """
    Scan thresholds in [0, 1] to find best F1.
    Returns: (best_f1, best_threshold)
    """
    best_f1 = -1.0
    best_t = 0.5

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    for t in thresholds:
        _, _, f1 = precision_recall_f1(y_true, y_prob, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_f1, best_t
