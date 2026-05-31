"""Memory-safe data loading utilities."""
from pathlib import Path
import json
from typing import Iterable, Dict, Any, List
import ijson


def find_file(filename: str, base_dir: str | Path) -> Path:
    base_dir = Path(base_dir)
    matches = list(base_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {base_dir}")
    return sorted(matches, key=lambda p: ("training" not in str(p).lower(), len(str(p))))[0]


def get_first_n_route_ids(json_path: str | Path, n: int = 100) -> List[str]:
    route_ids = []
    with open(json_path, "rb") as f:
        for key, _value in ijson.kvitems(f, ""):
            route_ids.append(key)
            if len(route_ids) >= n:
                break
    return route_ids


def load_selected_routes(json_path: str | Path, selected_route_ids: Iterable[str]) -> Dict[str, Any]:
    selected = set(selected_route_ids)
    loaded = {}
    with open(json_path, "rb") as f:
        for key, value in ijson.kvitems(f, ""):
            if key in selected:
                loaded[key] = value
                if len(loaded) == len(selected):
                    break
    return loaded


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
