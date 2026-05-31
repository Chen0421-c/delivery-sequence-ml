"""Preprocessing and validation utilities for single-route stop sequencing."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


RouteDict = Dict[str, Any]


def _sequence_mapping(seq_obj: Any) -> Dict[str, int]:
    """Normalize Amazon sequence records to ``{stop_id: visit_order}``."""
    if isinstance(seq_obj, dict) and "actual" in seq_obj and isinstance(seq_obj["actual"], dict):
        seq_obj = seq_obj["actual"]
    if isinstance(seq_obj, dict):
        return {str(stop_id): int(order) for stop_id, order in seq_obj.items()}
    if isinstance(seq_obj, list):
        return {str(stop_id): order for order, stop_id in enumerate(seq_obj)}
    raise ValueError("Unsupported actual sequence format")


def get_actual_sequence(route_id: str, actual_sequences: RouteDict) -> List[str]:
    """Return the historical stop list ordered by actual visit order."""
    mapping = _sequence_mapping(actual_sequences[route_id])
    return [stop_id for stop_id, _order in sorted(mapping.items(), key=lambda item: item[1])]


def get_route_stops(route_id: str, route_data: RouteDict) -> Dict[str, Any]:
    """Return the stop dictionary for a route, normalized to string IDs."""
    stops = route_data[route_id].get("stops", {})
    return {str(stop_id): info for stop_id, info in stops.items()}


def get_depot_stop_id(route_id: str, route_data: RouteDict, actual_sequences: RouteDict | None = None) -> str:
    """Identify the station/depot stop for a route.

    The Amazon data generally labels the station stop with type ``Station``.
    If absent, use the first stop in the actual sequence, then the first route
    stop as a final fallback.
    """
    stops = get_route_stops(route_id, route_data)
    for stop_id, info in stops.items():
        stop_type = str(info.get("type", info.get("stop_type", ""))).lower()
        if stop_type in {"station", "depot"}:
            return stop_id
    if actual_sequences and route_id in actual_sequences:
        sequence = get_actual_sequence(route_id, actual_sequences)
        if sequence:
            return sequence[0]
    if stops:
        return next(iter(stops))
    raise ValueError(f"Route {route_id} has no stops")


def package_count_for_stop(route_id: str, stop_id: str, package_data: RouteDict | None) -> int:
    """Count packages associated with one stop."""
    if not package_data or route_id not in package_data:
        return 0
    stop_packages = package_data[route_id].get(stop_id, {})
    if isinstance(stop_packages, dict):
        return len(stop_packages)
    if isinstance(stop_packages, list):
        return len(stop_packages)
    return 0


def validate_route(
    route_id: str,
    route_data: RouteDict,
    actual_sequences: RouteDict,
    package_data: RouteDict,
    travel_times: RouteDict,
) -> Tuple[bool, List[str]]:
    """Check route, stop and travel-time consistency for one route."""
    errors: List[str] = []
    for name, source in {
        "route_data": route_data,
        "actual_sequences": actual_sequences,
        "package_data": package_data,
        "travel_times": travel_times,
    }.items():
        if route_id not in source:
            errors.append(f"missing {name}")

    if errors:
        return False, errors

    route_stops = set(get_route_stops(route_id, route_data))
    if not route_stops:
        errors.append("missing route stops")
        return False, errors

    try:
        actual_sequence = get_actual_sequence(route_id, actual_sequences)
    except Exception as exc:  # noqa: BLE001 - validation should report malformed rows.
        return False, [f"invalid actual sequence: {exc}"]

    actual_stops = set(actual_sequence)
    if not actual_stops:
        errors.append("missing actual sequence stops")
    if actual_stops != route_stops:
        missing_from_sequence = route_stops - actual_stops
        missing_from_route = actual_stops - route_stops
        if missing_from_sequence:
            errors.append(f"{len(missing_from_sequence)} route stops missing from sequence")
        if missing_from_route:
            errors.append(f"{len(missing_from_route)} sequence stops missing from route")

    tt = travel_times.get(route_id, {})
    tt_rows = {str(stop_id) for stop_id in tt.keys()}
    if not route_stops.issubset(tt_rows):
        errors.append(f"{len(route_stops - tt_rows)} route stops missing travel-time rows")
    for stop_id in route_stops:
        row = tt.get(stop_id, {})
        if not isinstance(row, dict):
            errors.append(f"travel-time row for {stop_id} is not a mapping")
            continue
        row_stops = {str(candidate) for candidate in row.keys()}
        missing_columns = route_stops - row_stops
        if missing_columns:
            errors.append(f"travel-time row {stop_id} misses {len(missing_columns)} stops")
            break

    return not errors, errors


def filter_valid_routes(
    route_data: RouteDict,
    actual_sequences: RouteDict,
    package_data: RouteDict,
    travel_times: RouteDict,
) -> Tuple[List[str], pd.DataFrame]:
    """Return valid route IDs and a validation report for invalid routes."""
    candidate_ids = sorted(set(route_data) | set(actual_sequences) | set(package_data) | set(travel_times))
    valid_route_ids: List[str] = []
    report_rows: List[Dict[str, str]] = []
    for route_id in candidate_ids:
        is_valid, errors = validate_route(route_id, route_data, actual_sequences, package_data, travel_times)
        if is_valid:
            valid_route_ids.append(route_id)
        else:
            report_rows.append({"route_id": route_id, "reason": "; ".join(errors)})
    return valid_route_ids, pd.DataFrame(report_rows)


def route_to_stop_table(
    route_id: str,
    route_data: RouteDict,
    actual_sequences: RouteDict,
    package_data: RouteDict | None = None,
) -> pd.DataFrame:
    """Extract stop-level features for one route."""
    stops = get_route_stops(route_id, route_data)
    sequence_mapping = _sequence_mapping(actual_sequences[route_id])
    depot_id = get_depot_stop_id(route_id, route_data, actual_sequences)
    rows = []
    for stop_id, info in stops.items():
        rows.append(
            {
                "route_id": route_id,
                "stop_id": stop_id,
                "actual_order": sequence_mapping.get(stop_id),
                "is_depot": stop_id == depot_id,
                "lat": info.get("lat", np.nan),
                "lng": info.get("lng", np.nan),
                "zone_id": info.get("zone_id"),
                "stop_type": info.get("type", info.get("stop_type")),
                "package_count": package_count_for_stop(route_id, stop_id, package_data),
            }
        )
    return pd.DataFrame(rows).sort_values("actual_order", na_position="last").reset_index(drop=True)


def build_stop_feature_table(
    route_ids: Iterable[str],
    route_data: RouteDict,
    actual_sequences: RouteDict,
    package_data: RouteDict | None = None,
) -> pd.DataFrame:
    """Concatenate stop feature tables for multiple valid routes."""
    frames = [route_to_stop_table(route_id, route_data, actual_sequences, package_data) for route_id in route_ids]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
