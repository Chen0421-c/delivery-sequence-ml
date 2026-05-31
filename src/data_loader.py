"""Memory-safe data loading utilities for the Amazon Last Mile dataset.

The Amazon files are JSON objects keyed by route ID.  ``travel_times.json`` is
large, so this module streams top-level route records with :mod:`ijson` and only
materializes the requested route IDs.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Iterable, List, Mapping

import ijson

N_ROUTES = 100


DATASET_FILES = {
    "route_data": "route_data.json",
    "actual_sequences": "actual_sequences.json",
    "package_data": "package_data.json",
    "travel_times": "travel_times.json",
}


def find_file(filename: str, base_dir: str | Path) -> Path:
    """Find a dataset file under ``base_dir`` without hard-coded paths."""
    base_dir = Path(base_dir).expanduser().resolve()
    matches = list(base_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {base_dir}")

    # Prefer canonical training folders when both sample/eval files are present.
    return sorted(matches, key=lambda p: ("training" not in str(p).lower(), len(str(p))))[0]


def resolve_dataset_paths(data_dir: str | Path) -> Dict[str, Path]:
    """Return paths for the four Stage 1 Amazon JSON files."""
    return {name: find_file(filename, data_dir) for name, filename in DATASET_FILES.items()}


def stream_route_ids(json_path: str | Path) -> Iterable[str]:
    """Yield top-level route IDs from a route-keyed JSON file using ijson."""
    with Path(json_path).open("rb") as f:
        for route_id, _route_obj in ijson.kvitems(f, ""):
            yield str(route_id)


def get_first_n_route_ids(json_path: str | Path, n: int = N_ROUTES) -> List[str]:
    """Read only the first ``n`` top-level route IDs from ``json_path``."""
    route_ids: List[str] = []
    for route_id in stream_route_ids(json_path):
        route_ids.append(route_id)
        if len(route_ids) >= n:
            break
    return route_ids


def load_selected_routes(json_path: str | Path, selected_route_ids: Iterable[str]) -> Dict[str, Any]:
    """Stream ``json_path`` and load only the selected top-level route records."""
    selected = {str(route_id) for route_id in selected_route_ids}
    loaded: Dict[str, Any] = {}
    if not selected:
        return loaded

    with Path(json_path).open("rb") as f:
        for route_id, value in ijson.kvitems(f, ""):
            route_id = str(route_id)
            if route_id in selected:
                loaded[route_id] = value
                if len(loaded) == len(selected):
                    break
    return loaded


def load_stage1_subset(data_dir: str | Path, n_routes: int = N_ROUTES) -> Dict[str, Dict[str, Any]]:
    """Load a memory-safe subset of the Amazon Last Mile dataset.

    Route IDs are selected from ``route_data.json`` and then those same IDs are
    streamed from ``actual_sequences.json``, ``package_data.json`` and
    ``travel_times.json``.  The full travel-time file is never loaded by
    default.
    """
    paths = resolve_dataset_paths(data_dir)
    route_ids = get_first_n_route_ids(paths["route_data"], n_routes)
    return {
        "route_data": load_selected_routes(paths["route_data"], route_ids),
        "actual_sequences": load_selected_routes(paths["actual_sequences"], route_ids),
        "package_data": load_selected_routes(paths["package_data"], route_ids),
        "travel_times": load_selected_routes(paths["travel_times"], route_ids),
    }


def save_json(obj: Mapping[str, Any], path: str | Path) -> None:
    """Save JSON with parent-directory creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
