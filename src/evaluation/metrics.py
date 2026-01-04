import torch
from collections import defaultdict

from ontology import go_ic
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

    Key behaviors:
    - Converts logits -> probs via sigmoid
    - For each threshold, builds predicted GO set per protein, applies ancestor propagation
    - Computes precision/recall aggregated over proteins
    - Returns best F1 (Fmax) and its threshold

    Why these changes (vs previous version):
    1) Correct IC-weighting logic:
       - If go_ic is provided, tp/fp/fn should be IC-weighted sums.
       - If go_ic is None, tp/fp/fn should be unweighted counts.
       The previous version had this reversed.

    2) Robustness:
       - Uses go_ic.get(go_id, 0.0) to avoid KeyError when an ontology term is missing IC.
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


    # Precompute propagated ture GO sets (CPU-side sets)
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


    best_f1 = 0.0
    best_t = 0.0

    use_ic = go_ic is not None

    for t in thresholds:
        tp = fp = fn = 0.0

        for i in range(N):
            scores = probs[i]

            # ---------- add Top-K to cut off the computation ----------
            if topk is not None:
                k = min(topk, scores.numel())
                top_idx = torch.topk(scores, k).indices.tolist()
            else:
                top_idx = range(C)
            # ---------------------------------

            pred_go = {
                idx2go[j] for j in top_idx if scores[j] >= t
            }
            pred_go = propagate_ancestors(pred_go, godag)

            true_go = true_go_sets[i]
            inter = pred_go & true_go
            pred_only = pred_go - true_go
            true_only = true_go - pred_go

            if not use_ic:
                # Unweighted counts
                tp += float(len(inter))
                fp += float(len(pred_only))
                fn += float(len(true_only))
            else:
                # IC-weighted sums (safe get)
                tp += sum(float(go_ic.get(g, 0.0)) for g in inter)
                fp += sum(float(go_ic.get(g, 0.0)) for g in pred_only)
                fn += sum(float(go_ic.get(g, 0.0)) for g in true_only)

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_f1, best_t

