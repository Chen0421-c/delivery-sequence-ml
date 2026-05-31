"""Folium route visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import folium


def plot_route_folium(
    route_id: str,
    sequence: List[str],
    route_data: Dict[str, Any],
    output_html_path: str | Path | None = None,
) -> folium.Map:
    """Create a simple stop-order map for one route sequence."""
    stops = route_data[route_id].get("stops", {})
    coords = []
    for stop_id in sequence:
        info = stops.get(stop_id, {})
        lat = info.get("lat")
        lng = info.get("lng")
        if lat is not None and lng is not None:
            coords.append((float(lat), float(lng), stop_id))
    if not coords:
        raise ValueError(f"No valid coordinates found for route {route_id}")

    center = [sum(point[0] for point in coords) / len(coords), sum(point[1] for point in coords) / len(coords)]
    route_map = folium.Map(location=center, zoom_start=12)
    line = []
    for order, (lat, lng, stop_id) in enumerate(coords):
        line.append([lat, lng])
        folium.Marker(
            [lat, lng],
            tooltip=f"{order}: {stop_id}",
            popup=f"Route: {route_id}<br>Order: {order}<br>Stop: {stop_id}",
        ).add_to(route_map)
    folium.PolyLine(line, weight=3, opacity=0.8).add_to(route_map)

    if output_html_path:
        output_html_path = Path(output_html_path)
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        route_map.save(str(output_html_path))
    return route_map
