"""Position bias and verbosity correlation."""

from __future__ import annotations


def count_tokens_simple(text: str) -> int:
    """Simple deterministic token count (words) for tests; swap to tiktoken in prod if desired."""
    return len(text.split()) if text else 0


def position_bias_rate(
    pairs: list[tuple[str, str, str]],
    run_comparative_judge: callable,
) -> float:
    """
    pairs: list of (question, response_a, response_b).
    run_comparative_judge(question, response_a, response_b) -> winner "A" or "B".
    Returns fraction of pairs where winner flips when order is swapped (B vs A).
    """
    if not pairs:
        return 0.0
    flips = 0
    for question, resp_a, resp_b in pairs:
        winner_ab = run_comparative_judge(question, resp_a, resp_b)
        winner_ba = run_comparative_judge(question, resp_b, resp_a)
        if winner_ab and winner_ba and winner_ab != winner_ba:
            flips += 1
    return flips / len(pairs)


def verbosity_correlation(
    lengths: list[int],
    scores: list[float],
) -> float:
    """Pearson correlation between response lengths (e.g. token counts) and scores."""
    n = len(lengths)
    if n != len(scores) or n < 2:
        return 0.0
    mean_l = sum(lengths) / n
    mean_s = sum(scores) / n
    cov = sum((l - mean_l) * (s - mean_s) for l, s in zip(lengths, scores))
    var_l = sum((l - mean_l) ** 2 for l in lengths)
    var_s = sum((s - mean_s) ** 2 for s in scores)
    if var_l == 0 or var_s == 0:
        return 0.0
    return cov / (var_l * var_s) ** 0.5
