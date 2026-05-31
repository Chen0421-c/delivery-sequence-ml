"""Stage 1 baseline route sequencing methods."""
from __future__ import annotations

from typing import Dict, List


def calculate_sequence_travel_time(sequence: List[str], travel_time_matrix: Dict[str, Dict[str, float]]) -> float:
    """Calculate path travel time without returning to the depot."""
    return sum(float(travel_time_matrix.get(a, {}).get(b, 0.0)) for a, b in zip(sequence[:-1], sequence[1:]))


def nearest_neighbour_sequence(stop_ids: List[str], depot_id: str, travel_time_matrix: Dict[str, Dict[str, float]]) -> List[str]:
    """Generate a single-route nearest-neighbour stop sequence.

    Starting from ``depot_id``, repeatedly choose the unvisited stop with the
    smallest travel time from the current stop.
    """
    if depot_id not in stop_ids:
        raise ValueError(f"Depot {depot_id!r} is not in the stop list")

    unvisited = set(stop_ids)
    unvisited.remove(depot_id)
    sequence = [depot_id]
    current = depot_id

    while unvisited:
        next_stop = min(unvisited, key=lambda stop_id: float(travel_time_matrix[current][stop_id]))
        sequence.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop

    return sequence
