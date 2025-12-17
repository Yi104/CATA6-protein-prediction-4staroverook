import torch
from collections import defaultdict
from src.ontology.propagation import propagate_ancestors
from src.ontology.go_ic import compute_go_ic

def compute_fmax_hierarchical(
    logits: torch.Tensor,
    targets: torch.Tensor,
    idx2go: list,
    godag,
    go_ic = None,
    thresholds=None, # for simple base line test. later none.
    eps: float = 1e-8,
    max_proteins =None, # or None
    topk =500 # add one variable 500 for full evaluation
):
    """
    Hierarchy-aware (but unweighted) Fmax, protein-centric.
    This is the correct next baseline toward CAFA.
    """

    if thresholds is None:
        thresholds = torch.linspace(0.01, 0.95, 19) # can be updated later

    probs = torch.sigmoid(logits)

    N, C = probs.shape # N is number of validation proteins.
    # add this condition to limit the protein num in metrics to save computation runtime
    if max_proteins is not None:
        N = min(N, max_proteins)
        probs = probs[:N]
        targets = targets[:N]

    best_f1 = 0.0
    best_t = 0.0

    # Precompute true GO sets per protein
    true_go_sets = []
    for i in range(N):
        go_set = {
            idx2go[j]
            for j in range(C)
            if targets[i, j] > 0
        }
        true_go_sets.append(
            propagate_ancestors(go_set, godag)
        )

    for t in thresholds:
        tp = fp = fn = 0.0

        for i in range(N):
            scores = probs[i]

            # ---------- ad Top-K to cut off the computation ----------
            if topk is not None:
                k = min(topk, scores.numel())
                top_idx = torch.topk(scores, k).indices.tolist()
            else:
                top_idx = range(C)
            # ---------------------------------

            pred_go = {
                idx2go[j]
                for j in top_idx
                if scores[j] >= t
            }
            pred_go = propagate_ancestors(pred_go, godag)

            true_go = true_go_sets[i]

            if go_ic is not None:
                tp += len(pred_go & true_go)
                fp += len(pred_go - true_go)
                fn += len(true_go - pred_go)
            else:
                # ic_weighted
                tp += sum(go_ic[g] for g in pred_go & true_go)
                fp += sum(go_ic[g] for g in pred_go - true_go)
                fn += sum(go_ic[g] for g in true_go - pred_go)

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_f1, best_t

# def compute_microf1_debug(
#     logits: torch.Tensor,
#     targets: torch.Tensor,
#     thresholds=None,
#     eps: float = 1e-8,
# ):
#     """
#     Compute Fmax for multi-label classification.
#     # this is only to test the pipeline, not the final fmax should be used.
#
#     Parameters
#     ----------
#     logits : torch.Tensor
#         Shape (N, C), raw model outputs (before sigmoid).
#     targets : torch.Tensor
#         Shape (N, C), binary ground-truth labels {0,1}.
#     thresholds : iterable of float, optional
#         Thresholds to evaluate. Default: torch.linspace(0.01, 0.99, 99).
#     eps : float
#         Numerical stability constant.
#
#     Returns
#     -------
#     fmax : float
#         Maximum F1 score across thresholds.
#     best_threshold : float
#         Threshold achieving Fmax.
#     """
#
#     if thresholds is None:
#         thresholds = torch.linspace(0.01, 0.99, 99)
#
#     probs = torch.sigmoid(logits)
#
#     best_f1 = 0.0
#     best_t = 0.0
#
#     # Flatten for micro-F1
#     y_true = targets.view(-1)
#     probs = probs.view(-1)
#
#     for t in thresholds:
#         y_pred = (probs >= t).float()
#
#         tp = (y_pred * y_true).sum()
#         fp = (y_pred * (1 - y_true)).sum()
#         fn = ((1 - y_pred) * y_true).sum()
#
#         precision = tp / (tp + fp + eps)
#         recall = tp / (tp + fn + eps)
#
#         f1 = 2 * precision * recall / (precision + recall + eps)
#
#         if f1 > best_f1:
#             best_f1 = f1.item()
#             best_t = t.item()
#
#     return best_f1, best_t
