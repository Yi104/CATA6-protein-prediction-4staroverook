# src/ontology/propagation.py
from goatools.obo_parser import GODag

_ancestor_cache = {}

def propagate_ancestors(go_set, godag):
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
                _ancestor_cache[go_id] = godag[go_id].get_all_parents()
            else:
                _ancestor_cache[go_id] = set()

        expanded |= _ancestor_cache[go_id]


    return expanded
