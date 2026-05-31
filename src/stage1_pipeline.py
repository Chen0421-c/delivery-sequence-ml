"""End-to-end Stage 1 MVP pipeline.

Run from the repository root, for example:

    python -m src.stage1_pipeline --data-dir data --n-routes 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .baseline import nearest_neighbour_sequence
from .data_loader import N_ROUTES, load_stage1_subset
from .evaluation import evaluate_sequence
from .ml_model import (
    build_next_stop_training_samples,
    generate_rf_sequence,
    next_stop_accuracy,
    route_level_train_test_split,
    train_random_forest_model,
)
from .preprocessing import build_stop_feature_table, filter_valid_routes, get_actual_sequence, get_depot_stop_id, get_route_stops
from .visualization import plot_route_folium


def run_stage1(
    data_dir: str | Path,
    n_routes: int = N_ROUTES,
    results_dir: str | Path = "results",
    figures_dir: str | Path = "figures",
    test_size: float = 0.2,
    random_state: int = 42,
    max_maps: int = 1,
) -> Dict[str, Any]:
    """Run the full Stage 1 MVP on a route subset."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_stage1_subset(data_dir, n_routes=n_routes)
    route_data = dataset["route_data"]
    actual_sequences = dataset["actual_sequences"]
    package_data = dataset["package_data"]
    travel_times = dataset["travel_times"]

    valid_route_ids, invalid_report = filter_valid_routes(route_data, actual_sequences, package_data, travel_times)
    invalid_report.to_csv(results_dir / "invalid_routes.csv", index=False)
    if not valid_route_ids:
        raise ValueError("No valid routes found after preprocessing checks")

    stop_features = build_stop_feature_table(valid_route_ids, route_data, actual_sequences, package_data)
    stop_features.to_csv(results_dir / "stop_features.csv", index=False)

    train_route_ids, test_route_ids = route_level_train_test_split(valid_route_ids, test_size=test_size, random_state=random_state)
    if not test_route_ids:
        test_route_ids = train_route_ids

    train_samples = build_next_stop_training_samples(train_route_ids, route_data, actual_sequences, package_data, travel_times)
    test_samples = build_next_stop_training_samples(test_route_ids, route_data, actual_sequences, package_data, travel_times)
    train_samples.to_csv(results_dir / "train_samples.csv", index=False)
    test_samples.to_csv(results_dir / "test_samples.csv", index=False)

    model = train_random_forest_model(train_samples, random_state=random_state)
    candidate_accuracy = next_stop_accuracy(model, test_samples)

    rows: List[Dict[str, Any]] = []
    map_count = 0
    for route_id in test_route_ids:
        actual_sequence = get_actual_sequence(route_id, actual_sequences)
        depot_id = get_depot_stop_id(route_id, route_data, actual_sequences)
        stop_ids = list(get_route_stops(route_id, route_data))
        matrix = travel_times[route_id]

        nn_sequence = nearest_neighbour_sequence(stop_ids, depot_id, matrix)
        rf_sequence = generate_rf_sequence(route_id, model, route_data, actual_sequences, package_data, travel_times)

        rows.append(evaluate_sequence(route_id, "actual_historical", actual_sequence, actual_sequence, matrix))
        rows.append(evaluate_sequence(route_id, "nearest_neighbour", actual_sequence, nn_sequence, matrix))
        rows.append(evaluate_sequence(route_id, "random_forest", actual_sequence, rf_sequence, matrix))

        if map_count < max_maps:
            plot_route_folium(route_id, actual_sequence, route_data, figures_dir / f"{route_id}_actual.html")
            plot_route_folium(route_id, nn_sequence, route_data, figures_dir / f"{route_id}_nearest_neighbour.html")
            plot_route_folium(route_id, rf_sequence, route_data, figures_dir / f"{route_id}_random_forest.html")
            map_count += 1

    evaluation = pd.DataFrame(rows)
    evaluation["next_stop_prediction_accuracy"] = candidate_accuracy
    evaluation.to_csv(results_dir / "evaluation.csv", index=False)

    summary = evaluation.groupby("method", as_index=False).mean(numeric_only=True)
    summary.to_csv(results_dir / "evaluation_summary.csv", index=False)

    return {
        "valid_route_ids": valid_route_ids,
        "train_route_ids": train_route_ids,
        "test_route_ids": test_route_ids,
        "invalid_report": invalid_report,
        "stop_features": stop_features,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "evaluation": evaluation,
        "summary": summary,
        "model": model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 single-route delivery sequencing MVP.")
    parser.add_argument("--data-dir", default="data", help="Directory containing or nesting the Amazon JSON dataset files.")
    parser.add_argument("--n-routes", type=int, default=N_ROUTES, help="Number of routes to stream-load from the dataset.")
    parser.add_argument("--results-dir", default="results", help="Directory for CSV outputs.")
    parser.add_argument("--figures-dir", default="figures", help="Directory for Folium HTML maps.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Route-level test split fraction.")
    parser.add_argument("--max-maps", type=int, default=1, help="Number of test routes to map for each method.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_stage1(
        data_dir=args.data_dir,
        n_routes=args.n_routes,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        test_size=args.test_size,
        max_maps=args.max_maps,
    )
    print(outputs["summary"])


if __name__ == "__main__":
    main()
