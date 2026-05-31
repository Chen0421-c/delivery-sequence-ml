"""Visualization utilities."""
from pathlib import Path
from typing import Dict, Any, List
import folium


def plot_route_folium(route_id: str, sequence: List[str], route_data: Dict[str, Any], output_html_path: str | Path | None = None):
    stops = route_data[route_id].get('stops', {})
    coords = []
    for stop_id in sequence:
        info = stops.get(stop_id, {})
        if info.get('lat') is not None and info.get('lng') is not None:
            coords.append((float(info['lat']), float(info['lng']), stop_id))
    if not coords:
        raise ValueError('No valid coordinates found.')
    center = [sum(x[0] for x in coords)/len(coords), sum(x[1] for x in coords)/len(coords)]
    m = folium.Map(location=center, zoom_start=12)
    line = []
    for i, (lat, lng, stop_id) in enumerate(coords):
        line.append([lat, lng])
        folium.Marker([lat, lng], tooltip=f'{i}: {stop_id}', popup=f'{i}: {stop_id}').add_to(m)
    folium.PolyLine(line, weight=3).add_to(m)
    if output_html_path:
        Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_html_path))
    return m
