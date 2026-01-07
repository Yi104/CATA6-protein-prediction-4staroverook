
"""
    ---------------------------------------------------------------------------
    VERSION TRACKING: IMPROVEMENTS
    ---------------------------------------------------------------------------
    [ALGORITHMIC IMPROVEMENT: THE FMAX CALCULATION]
    - V2 Fmax was likely calculated using 'Micro-averaging'
    (summing all True Positives across all proteins, then calculating one F1).

    In this version (v3), we implement 'Protein-Centric Macro-averaging':
    1. We calculate Precision and Recall for EVERY protein individually at each threshold.
    2. We average those Precisions and Recalls across the dataset.
    3. We find the threshold that maximizes this average.

     [CORE LOGIC CORRECTION: HIERARCHY]
    - v1: Likely performed 'flat' evaluation (comparing predicted indices directly to targets).
    - v3: Performs 'Hierarchical' evaluation. It uses 'godag' to find all ancestors.
      Why: Flat evaluation severely penalizes models for near-misses in the hierarchy.

    [EFFICIENCY: ANCESTOR CACHING]
    - v1: Recalculated the GO tree path for every protein at every threshold.
    - v3: Builds an 'Ancestor Cache'. We identify unique terms in the batch and
      pre-calculate their paths once. This reduces graph traversal overhead by ~90%.

    [ALGORITHMIC: TOP-K PARTITIONING]
    - v2: Used `torch.sort(logits)` which is O(C log C) for ~30k classes.
    - v3: Uses `np.argpartition` to find the top_k in O(C) linear time.
      Why: We don't need to know the order of the thousands of terms the model
      is sure are 'false'. We only need the most confident candidates.

    [MEMORY: BATCH TRUNCATION]
    - v3: Added `max_proteins`. Large-scale validation (e.g. Swissprot) can exceed
      available system RAM when expanding hierarchies. This allows safe sampling.
"""

import torch
from collections import defaultdict
import numpy as np

from src.ontology.propagation import propagate_ancestors
from typing import Dict, List, Optional, Set, Tuple

def compute_fmax_hierarchical(
    logits: torch.Tensor,
    targets: torch.Tensor,
    idx2go: list[str],
    godag,
    go_ic: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str,float]] = None, # for simple baseline test. later none.
    max_proteins: Optional[int] = None, # or None
    top_k =500, # add one variable 500 for full evaluation
    return_best_threshold: bool = True,

):
    """
    Overview:
     ---------------------------------
    This function evaluates multi-label protein function predictions using the CAFA
    (Critical Assessment of Protein Function Annotation) standard.

    1. Hierarchical Propagation: Since Gene Ontology (GO) is a Directed Acyclic Graph,
       if a protein is labeled with 'Nucleus', it is implicitly labeled with
       'Cellular Component'. This function 'propagates' labels up the tree to
       ensure predictions and ground truths are biologically consistent.

    2. Fmax Calculation: It iterates through different confidence thresholds to find
        the maximum Harmonic Mean of Precision and Recall (F-score).

    3. Semantic Weighting: If Information Content (IC) is provided, it calculates
       weighted metrics where predicting rare/specific terms is worth more than
       predicting common/general terms.

     Returns:
        dict with:
          - fmax_unweighted
          - best_t_unweighted (if return_best_threshold=True)
          - fmax_weighted (if go_ic provided)
          - best_t_weighted (if go_ic provided and return_best_threshold=True)

    """

    # --- 1. PRE-PROCESSING & SUB-SAMPLING ---
    # Limits memory usage if we are just doing a quick validation check.
    if max_proteins is not None and max_proteins < logits.shape[0]:
        logits = logits[:max_proteins]
        targets = targets[:max_proteins]

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    n_proteins, n_classes = probs.shape

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    else:
        thresholds = np.asarray(thresholds)

    # --- 2. GROUND TRUTH PROPAGATION ---
    # What it does: Converts sparse binary vectors into sets of GO IDs including all
    # parents. This is the "True" biological state of the protein.
    gt_sets: List[Set[str]] = []
    gt_ic_sums: Optional[List[float]] = [] if go_ic else None

    for i in range(n_proteins):
        direct_indices = np.where(targets_np[i] > 0)[0]
        full_gt: Set[str] = set()

        for idx in direct_indices:
            go_id = idx2go[int(idx)]
            # Always include self, even if not in godag (consistency with pred handling)
            full_gt.add(go_id)
            if go_id in godag:
                # goatools: transitive ancestor closure
                full_gt.update(godag[go_id].get_all_parents())

        gt_sets.append(full_gt)
        if go_ic:
            gt_ic_sums.append(sum(float(go_ic.get(go, 0.0)) for go in full_gt))

    # --- 3. OPTIMIZED CANDIDATE SELECTION ---
    # What it does: Instead of checking 30,000 labels for every threshold,
    # we only look at the 'top_k' most likely candidates.
    top_partition = np.argpartition(probs, -top_k, axis=1)[:, -top_k:]
    row_indices = np.arange(n_proteins)[:, None]
    top_probs = probs[row_indices, top_partition]
    # Sort only the small top_k subset
    sorted_sub_indices = np.argsort(-top_probs, axis=1)
    top_indices = top_partition[row_indices, sorted_sub_indices] # (n_proteins, top_k)

    # --- 4. THE ANCESTOR CACHE (Optimization) ---
    # **** Maps a single GO term to its entire lineage once. *****
    unique_top_indices = np.unique(top_indices)
    ancestor_cache_idx: Dict[int, Set[str]] = {}
    for cls_idx in unique_top_indices:
        cls_idx_int = int(cls_idx)
        go_id = idx2go[cls_idx_int]
        if go_id in godag:
            anc = set(godag[go_id].get_all_parents())
            anc.add(go_id)
            ancestor_cache_idx[cls_idx_int] = anc
        else:
            ancestor_cache_idx[cls_idx_int] = {go_id}

    # --- 4b) IC cache to speed up weighted mode ---
    ic_cache: Optional[Dict[str, float]] = None
    if go_ic:
        ic_cache = defaultdict(float)
        # Cache all GO terms that can appear in pred or GT
        for s in ancestor_cache_idx.values():
            for go in s:
                ic_cache[go] = float(go_ic.get(go, 0.0))
        for s in gt_sets:
            for go in s:
                # only fills missing keys cheaply
                if go not in ic_cache:
                    ic_cache[go] = float(go_ic.get(go, 0.0))

    # --- 5. THE EVALUATION LOOP ---
    def evaluate_fmax(use_weight: bool):
        best_f1  = 0.0
        best_t = float(thresholds[0])

        for t in thresholds:
            p_list = []
            r_list = []

            for i in range(n_proteins):
                current_top = top_indices[i]  # (top_k,)
                # filter by threshold on raw probs
                mask = probs[i, current_top] >= t
                pred_cls = current_top[mask]

                if pred_cls.size == 0:
                    # CAFA rule: precision not counted for empty prediction; recall=0
                    r_list.append(0.0)
                    continue

                # union of ancestor sets for predicted classes
                pred_go: Set[str] = set()
                for cls_idx in pred_cls:
                    pred_go.update(ancestor_cache_idx[int(cls_idx)])

                true_go = gt_sets[i]
                inter = pred_go.intersection(true_go)

                if not use_weight:
                    tp = float(len(inter))
                    total_pred = float(len(pred_go))
                    total_true = float(len(true_go))
                else:
                    # Use cached IC values
                    tp = float(sum(ic_cache[g] for g in inter))
                    total_pred = float(sum(ic_cache[g] for g in pred_go))
                    total_true = float(gt_ic_sums[i])

                # per-protein P/R
                p_i = tp / total_pred if total_pred > 0.0 else 0.0
                r_i = tp / total_true if total_true > 0.0 else 0.0

                p_list.append(p_i)
                r_list.append(r_i)

            avg_p = float(np.mean(p_list)) if p_list else 0.0
            avg_r = float(np.mean(r_list)) if r_list else 0.0

            if (avg_p + avg_r) > 0.0:
                f1 = (2.0 * avg_p * avg_r) / (avg_p + avg_r)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)

        return best_f1, best_t



    # Final execution
    out = {}
    f_unw, t_unw = evaluate_fmax(use_weight=False)
    out["fmax_unweighted"] = f_unw
    if return_best_threshold:
        out["best_t_unweighted"] = t_unw

    if go_ic:
        f_w, t_w = evaluate_fmax(use_weight=True)
        out["fmax_weighted"] = f_w
        if return_best_threshold:
            out["best_t_weighted"] = t_w

    return out
