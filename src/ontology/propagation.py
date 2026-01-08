# src/ontology/propagation.py
"""
# CHANGE LOG (2026-01-07)
# This function performs ancestor closure over the GO DAG.
# In this project, it is used selectively:
#
# - Used for IC (Information Content) computation to obtain
#   ontology-consistent term frequencies.
# - NOT used during inference or submission, to avoid propagating
#   incorrect predictions to higher-level GO terms.
#
# Optionally, ontology root terms (MF/BP/CC) can be dropped to stabilize
# downstream statistics such as IC distributions.
#
# Cache stores parents only (excluding self); returned set includes self.

"""
from goatools.obo_parser import GODag

_ancestor_cache = {} # cache: go_id -> set(parents) excluding self
GO_ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}  # MF/BP/CC roots

def propagate_ancestors(go_set, godag,drop_roots: bool = False):
    """
    Expand a set of GO terms to include all ancestors
    using the GO DAG.

    Parameters
    ----------
    go_set : set[str]
        GO terms predicted or annotated for one protein.
    godag : goatools.obo_parser.GODag

    Returns
    -------
    expanded_set : set[str]
        GO terms including ancestors.
    """
    expanded = set(go_set)

    for go_id in go_set:
        if go_id not in _ancestor_cache:
            if go_id in godag:
                _ancestor_cache[go_id] = set(godag[go_id].get_all_parents())
            else:
                _ancestor_cache[go_id] = set()

        expanded |= _ancestor_cache[go_id]
    if drop_roots:
        expanded -= GO_ROOTS


    return expanded
