from goatools.obo_parser import GODag
import numpy as np
from goatools.obo_parser import GODag
from src.ontology.propagation import propagate_ancestors


def load_go_ontology(obo_path):
    """
    Load GO ontology and return go_id -> ontology (MF/BP/CC)
    """
    godag = GODag(obo_path)

    go2ont = {}
    for go_id, term in godag.items():
        ns = term.namespace
        if ns == "molecular_function":
            go2ont[go_id] = "MF"
        elif ns == "biological_process":
            go2ont[go_id] = "BP"
        elif ns == "cellular_component":
            go2ont[go_id] = "CC"

    return go2ont


def build_ontology_index(idx2go, go2ont):
    """
    Given idx2go list and go2ont mapping,
    return index arrays for MF, BP, CC
    """
    mf_idx = []
    bp_idx = []
    cc_idx = []

    for i, go_id in enumerate(idx2go):
        ont = go2ont.get(go_id)
        if ont == "MF":
            mf_idx.append(i)
        elif ont == "BP":
            bp_idx.append(i)
        elif ont == "CC":
            cc_idx.append(i)
        else:
            raise ValueError(f"GO term {go_id} missing ontology")

    return (
        np.array(mf_idx, dtype=int),
        np.array(bp_idx, dtype=int),
        np.array(cc_idx, dtype=int),
    )
