"""Evaluation metrics."""
from typing import List, Dict
import numpy as np
from scipy.stats import kendalltau, spearmanr


def route_travel_time(sequence: List[str], travel_time_matrix: Dict[str, Dict[str, float]]) -> float:
    return sum(float(travel_time_matrix.get(a, {}).get(b, 0.0)) for a, b in zip(sequence[:-1], sequence[1:]))


def percentage_gap(candidate_value: float, reference_value: float) -> float:
    return np.nan if reference_value == 0 else ((candidate_value - reference_value) / reference_value) * 100.0


def _rank(sequence: List[str]) -> Dict[str, int]:
    return {x: i for i, x in enumerate(sequence)}


def kendall_tau_similarity(actual_sequence: List[str], predicted_sequence: List[str]) -> float:
    common = [s for s in actual_sequence if s in set(predicted_sequence)]
    if len(common) < 2: return np.nan
    ar, pr = _rank(actual_sequence), _rank(predicted_sequence)
    return float(kendalltau([ar[s] for s in common], [pr[s] for s in common]).correlation)


def spearman_similarity(actual_sequence: List[str], predicted_sequence: List[str]) -> float:
    common = [s for s in actual_sequence if s in set(predicted_sequence)]
    if len(common) < 2: return np.nan
    ar, pr = _rank(actual_sequence), _rank(predicted_sequence)
    return float(spearmanr([ar[s] for s in common], [pr[s] for s in common]).correlation)


def edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    m, n = len(seq_a), len(seq_b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if seq_a[i-1] == seq_b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]
