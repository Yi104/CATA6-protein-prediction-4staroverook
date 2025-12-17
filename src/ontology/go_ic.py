import math
from collections import Counter

def compute_go_ic(label_dict: dict):
    """
    label_dict: {protein_id: set(go_terms)}
    return: dict {go_id: ic_value}
    """
    term_count = Counter()
    total_proteins = len(label_dict)

    for go_set in label_dict.values():
        for go in go_set:
            term_count[go] += 1


    go_ic = {}
    for go, cnt in term_count.items():
        p = cnt / total_proteins
        go_ic[go] = -math.log(p)

    return go_ic
