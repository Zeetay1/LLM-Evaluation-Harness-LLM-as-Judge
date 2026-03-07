"""Cohen's kappa (and weighted kappa) for judge vs human agreement."""

from __future__ import annotations

from collections import Counter


def cohens_kappa(judge_scores: list[int | float], human_scores: list[int | float]) -> float:
    """
    Cohen's kappa for two raters (judge vs human). Scores are categorical (e.g. 1-5).
    Both lists must have the same length. Uses ordinal (weighted) agreement if possible.
    """
    n = len(judge_scores)
    if n != len(human_scores) or n == 0:
        return 0.0
    # Round to int for categories
    j = [int(round(s)) for s in judge_scores]
    h = [int(round(s)) for s in human_scores]
    categories = sorted(set(j) | set(h))
    if len(categories) < 2:
        return 1.0
    # Weighted kappa: weight by distance (linear weights for ordinal)
    k = len(categories)
    w = [[1.0 - abs(i - j) / (k - 1) if k > 1 else 1.0 for j in range(k)] for i in range(k)]
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    observed = 0.0
    expected = 0.0
    count_j = Counter()
    count_h = Counter()
    for a, b in zip(j, h):
        ia = cat_to_idx.get(a, 0)
        ib = cat_to_idx.get(b, 0)
        observed += w[ia][ib] if ia < len(w) and ib < len(w[0]) else 0.0
        count_j[a] = count_j.get(a, 0) + 1
        count_h[b] = count_h.get(b, 0) + 1
    observed /= n
    for a in categories:
        for b in categories:
            ia = cat_to_idx.get(a, 0)
            ib = cat_to_idx.get(b, 0)
            p_j = count_j.get(a, 0) / n
            p_h = count_h.get(b, 0) / n
            weight = w[ia][ib] if ia < len(w) and ib < len(w[0]) else 0.0
            expected += p_j * p_h * weight
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0
