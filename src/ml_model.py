"""Random Forest next-stop prediction for Stage 1 MVP."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .preprocessing import get_actual_sequence, get_depot_stop_id, get_route_stops, package_count_for_stop

FEATURE_COLUMNS = [
    "travel_time_current_to_candidate",
    "current_lat",
    "current_lng",
    "candidate_lat",
    "candidate_lng",
    "same_zone",
    "candidate_package_count",
    "route_stop_count",
    "current_position_ratio",
]


def _stop_float(info: Dict[str, Any], field: str) -> float:
    value = info.get(field, 0.0)
    return 0.0 if value is None or pd.isna(value) else float(value)


def make_candidate_features(
    route_id: str,
    current_stop: str,
    candidate_stop: str,
    current_position: int,
    route_stop_count: int,
    route_data: Dict[str, Any],
    package_data: Dict[str, Any],
    travel_times: Dict[str, Any],
) -> Dict[str, float]:
    """Build one feature row for a current-stop/candidate-stop pair."""
    stops = get_route_stops(route_id, route_data)
    current_info = stops.get(current_stop, {})
    candidate_info = stops.get(candidate_stop, {})
    matrix = travel_times[route_id]
    return {
        "travel_time_current_to_candidate": float(matrix[current_stop][candidate_stop]),
        "current_lat": _stop_float(current_info, "lat"),
        "current_lng": _stop_float(current_info, "lng"),
        "candidate_lat": _stop_float(candidate_info, "lat"),
        "candidate_lng": _stop_float(candidate_info, "lng"),
        "same_zone": float(str(current_info.get("zone_id")) == str(candidate_info.get("zone_id"))),
        "candidate_package_count": float(package_count_for_stop(route_id, candidate_stop, package_data)),
        "route_stop_count": float(route_stop_count),
        "current_position_ratio": float(current_position / max(route_stop_count - 1, 1)),
    }


def build_next_stop_training_samples(
    route_ids: Iterable[str],
    route_data: Dict[str, Any],
    actual_sequences: Dict[str, Any],
    package_data: Dict[str, Any],
    travel_times: Dict[str, Any],
) -> pd.DataFrame:
    """Convert actual routes into supervised candidate next-stop samples.

    Each row is a pair ``(current stop, candidate stop)``.  The label is 1 only
    when the candidate is the historical next stop.
    """
    rows: List[Dict[str, Any]] = []
    for route_id in route_ids:
        actual_sequence = get_actual_sequence(route_id, actual_sequences)
        route_stop_count = len(actual_sequence)
        for position in range(route_stop_count - 1):
            current_stop = actual_sequence[position]
            true_next_stop = actual_sequence[position + 1]
            for candidate_stop in actual_sequence[position + 1 :]:
                row = make_candidate_features(
                    route_id,
                    current_stop,
                    candidate_stop,
                    position,
                    route_stop_count,
                    route_data,
                    package_data,
                    travel_times,
                )
                row.update(
                    {
                        "route_id": route_id,
                        "current_stop": current_stop,
                        "candidate_stop": candidate_stop,
                        "label": int(candidate_stop == true_next_stop),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def route_level_train_test_split(
    route_ids: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split route IDs so candidate rows from the same route cannot leak."""
    if len(route_ids) < 2:
        return route_ids, []
    train_ids, test_ids = train_test_split(route_ids, test_size=test_size, random_state=random_state)
    return sorted(train_ids), sorted(test_ids)


def train_random_forest_model(
    samples: pd.DataFrame,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train the Stage 1 Random Forest binary next-stop classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(samples[FEATURE_COLUMNS], samples["label"])
    return model


def predict_next_stop_probabilities(model: RandomForestClassifier, samples: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities, handling degenerate models."""
    probabilities = model.predict_proba(samples[FEATURE_COLUMNS])
    if probabilities.shape[1] == 1:
        return np.zeros(len(samples))
    positive_index = list(model.classes_).index(1)
    return probabilities[:, positive_index]


def generate_rf_sequence(
    route_id: str,
    model: RandomForestClassifier,
    route_data: Dict[str, Any],
    actual_sequences: Dict[str, Any],
    package_data: Dict[str, Any],
    travel_times: Dict[str, Any],
) -> List[str]:
    """Iteratively build a full sequence from Random Forest next-stop scores."""
    stops = list(get_route_stops(route_id, route_data))
    depot_id = get_depot_stop_id(route_id, route_data, actual_sequences)
    unvisited = set(stops)
    unvisited.remove(depot_id)
    sequence = [depot_id]
    current_stop = depot_id

    while unvisited:
        position = len(sequence) - 1
        candidate_stop_ids = sorted(unvisited)
        rows = [
            make_candidate_features(
                route_id,
                current_stop,
                candidate_stop,
                position,
                len(stops),
                route_data,
                package_data,
                travel_times,
            )
            for candidate_stop in candidate_stop_ids
        ]
        candidates = pd.DataFrame(rows)
        probabilities = predict_next_stop_probabilities(model, candidates)
        best_idx = int(np.argmax(probabilities))
        next_stop = candidate_stop_ids[best_idx]
        sequence.append(next_stop)
        unvisited.remove(next_stop)
        current_stop = next_stop

    return sequence


def next_stop_accuracy(model: RandomForestClassifier, samples: pd.DataFrame) -> float:
    """Measure top-1 next-stop accuracy within each decision state.

    For every ``(route_id, current_stop)`` group, the candidate with the highest
    predicted probability is treated as the model's next-stop choice.  The
    prediction is correct when that candidate has label 1.
    """
    if samples.empty:
        return float("nan")

    scored = samples.copy()
    scored["probability"] = predict_next_stop_probabilities(model, scored)
    correct = []
    for _group_key, group in scored.groupby(["route_id", "current_stop"], sort=False):
        best_idx = group["probability"].idxmax()
        correct.append(int(scored.loc[best_idx, "label"] == 1))
    return float(np.mean(correct)) if correct else float("nan")
