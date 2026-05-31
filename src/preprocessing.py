"""Preprocessing utilities."""
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def get_actual_sequence(route_id: str, actual_sequences: Dict[str, Any]) -> List[str]:
    seq_obj = actual_sequences[route_id]
    if isinstance(seq_obj, dict):
        return [sid for sid, _order in sorted(seq_obj.items(), key=lambda x: x[1])]
    return list(seq_obj)


def get_depot_stop_id(route_id: str, route_data: Dict[str, Any]) -> str:
    for stop_id, info in route_data[route_id].get("stops", {}).items():
        if str(info.get("type", "")).lower() in {"station", "depot"}:
            return stop_id
    return next(iter(route_data[route_id].get("stops", {}).keys()))


def route_to_stop_table(route_id: str, route_data: Dict[str, Any], actual_sequences: Dict[str, Any], package_data: Dict[str, Any] | None = None) -> pd.DataFrame:
    route = route_data[route_id]
    seq_obj = actual_sequences[route_id]
    rows = []
    for stop_id, info in route.get("stops", {}).items():
        rows.append({
            "route_id": route_id,
            "stop_id": stop_id,
            "actual_order": seq_obj.get(stop_id) if isinstance(seq_obj, dict) else None,
            "lat": info.get("lat", np.nan),
            "lng": info.get("lng", np.nan),
            "stop_type": info.get("type"),
            "zone_id": info.get("zone_id"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("actual_order", na_position="last").reset_index(drop=True)
