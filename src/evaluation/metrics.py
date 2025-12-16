import torch


def compute_fmax(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds=None,
    eps: float = 1e-8,
):
    """
    Compute Fmax for multi-label classification.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C), raw model outputs (before sigmoid).
    targets : torch.Tensor
        Shape (N, C), binary ground-truth labels {0,1}.
    thresholds : iterable of float, optional
        Thresholds to evaluate. Default: torch.linspace(0.01, 0.99, 99).
    eps : float
        Numerical stability constant.

    Returns
    -------
    fmax : float
        Maximum F1 score across thresholds.
    best_threshold : float
        Threshold achieving Fmax.
    """

    if thresholds is None:
        thresholds = torch.linspace(0.01, 0.99, 99)

    probs = torch.sigmoid(logits)

    best_f1 = 0.0
    best_t = 0.0

    # Flatten for micro-F1
    y_true = targets.view(-1)
    probs = probs.view(-1)

    for t in thresholds:
        y_pred = (probs >= t).float()

        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        fn = ((1 - y_pred) * y_true).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)

        if f1 > best_f1:
            best_f1 = f1.item()
            best_t = t.item()

    return best_f1, best_t
