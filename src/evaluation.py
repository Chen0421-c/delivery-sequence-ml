"""Evaluation metrics for Stage 1 route sequence comparison."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .baseline import calculate_sequence_travel_time


def route_travel_time(sequence: List[str], travel_time_matrix: Dict[str, Dict[str, float]]) -> float:
    """Calculate total travel time for a stop sequence."""
    return calculate_sequence_travel_time(sequence, travel_time_matrix)


def percentage_gap(candidate_value: float, reference_value: float) -> float:
    """Percentage gap versus a reference value."""
    return float("nan") if reference_value == 0 else ((candidate_value - reference_value) / reference_value) * 100.0


def _rank(sequence: List[str]) -> Dict[str, int]:
    return {stop_id: position for position, stop_id in enumerate(sequence)}


def kendall_tau_similarity(actual_sequence: List[str], predicted_sequence: List[str]) -> float:
    """Kendall Tau rank similarity on common stops."""
    common = [stop_id for stop_id in actual_sequence if stop_id in set(predicted_sequence)]
    if len(common) < 2:
        return float("nan")
    actual_rank, predicted_rank = _rank(actual_sequence), _rank(predicted_sequence)
    result = kendalltau([actual_rank[s] for s in common], [predicted_rank[s] for s in common]).correlation
    return float(result) if result is not None else float("nan")


def spearman_similarity(actual_sequence: List[str], predicted_sequence: List[str]) -> float:
    """Spearman rank correlation on common stops."""
    common = [stop_id for stop_id in actual_sequence if stop_id in set(predicted_sequence)]
    if len(common) < 2:
        return float("nan")
    actual_rank, predicted_rank = _rank(actual_sequence), _rank(predicted_sequence)
    result = spearmanr([actual_rank[s] for s in common], [predicted_rank[s] for s in common]).correlation
    return float(result) if result is not None else float("nan")


def edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """Levenshtein edit distance between two stop-ID sequences."""
    m, n = len(seq_a), len(seq_b)
    previous = list(range(n + 1))
    for i in range(1, m + 1):
        current = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[n]


def evaluate_sequence(route_id: str, method: str, actual_sequence: List[str], sequence: List[str], travel_time_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float | str]:
    """Evaluate one generated sequence against the historical sequence."""
    actual_time = route_travel_time(actual_sequence, travel_time_matrix)
    sequence_time = route_travel_time(sequence, travel_time_matrix)
    return {
        "route_id": route_id,
        "method": method,
        "total_travel_time": sequence_time,
        "percentage_gap_vs_actual": percentage_gap(sequence_time, actual_time),
        "kendall_tau": kendall_tau_similarity(actual_sequence, sequence),
        "spearman": spearman_similarity(actual_sequence, sequence),
        "edit_distance": edit_distance(actual_sequence, sequence),
    }


def summarize_evaluations(rows: List[Dict[str, float | str]]) -> pd.DataFrame:
    """Build a dataframe from route-level evaluation records."""
    return pd.DataFrame(rows)
