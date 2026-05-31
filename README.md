# Road Freight Delivery Sequence Optimization using Machine Learning

This repository implements the **Stage 1 MVP** for the MSc research project:

**Road Freight Delivery Sequence Optimization using Machine Learning**

## Stage 1 scope

Stage 1 is intentionally narrow:

- **Problem:** single-route, single-vehicle delivery stop sequence optimization.
- **Dataset:** Amazon Last Mile Routing Research Challenge Dataset.
- **Baseline:** Nearest Neighbour only.
- **Machine learning model:** Random Forest next-stop prediction only.
- **Excluded from Stage 1:** multi-vehicle CVRPTW, OR-Tools, XGBoost, deep learning, and production dashboards.

## Repository layout

```text
src/
  data_loader.py       # memory-safe route-keyed JSON loading with ijson
  preprocessing.py     # consistency checks, cleaning, depot detection, stop features
  baseline.py          # Nearest Neighbour sequence generation
  ml_model.py          # Random Forest candidate next-stop model
  evaluation.py        # travel time, gap, rank similarity, edit distance metrics
  visualization.py     # Folium route maps
  stage1_pipeline.py   # end-to-end Stage 1 runner
notebooks/
  main_colab.ipynb     # Google Colab workflow
results/               # generated CSV outputs; only .gitkeep is tracked
figures/               # generated Folium HTML maps; only .gitkeep is tracked
```

The `data/` folder is ignored by Git because the Amazon dataset is large.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data files

Download the Amazon Last Mile Routing Research Challenge Dataset from its official source and place the JSON files under `data/` or another local folder:

```text
data/
  route_data.json
  actual_sequences.json
  package_data.json
  travel_times.json
```

The loader recursively searches the selected data directory, so nested folders are also supported.

## Run Stage 1 MVP

The default subset size is `N_ROUTES = 100`. The pipeline streams route records and does **not** load the full `travel_times.json` into memory.

```bash
python -m src.stage1_pipeline --data-dir data --n-routes 100
```

Outputs:

- `results/invalid_routes.csv` — routes removed because of missing sequences, missing stops, or incomplete travel-time records.
- `results/stop_features.csv` — stop-level features: latitude, longitude, zone ID, stop type, package count, depot indicator, and actual order.
- `results/train_samples.csv` and `results/test_samples.csv` — supervised current-stop/candidate-stop samples split at route level.
- `results/evaluation.csv` — route-level comparison of actual historical, Nearest Neighbour, and Random Forest sequences.
- `results/evaluation_summary.csv` — average metrics by method.
- `figures/*_actual.html`, `figures/*_nearest_neighbour.html`, `figures/*_random_forest.html` — Folium route maps.

## Preprocessing and cleaning

Stage 1 validates each selected route before modelling:

1. Confirms the route ID exists across `route_data`, `actual_sequences`, `package_data`, and `travel_times`.
2. Confirms route stops match historical actual sequence stops.
3. Confirms the travel-time matrix contains rows and columns for every route stop.
4. Removes invalid routes from the working subset.
5. Identifies the depot/station stop from stop type, falling back to the first actual stop if necessary.
6. Extracts stop-level features: `lat`, `lng`, `zone_id`, `stop_type`, and `package_count`.
7. Converts actual sequences into ordered stop lists.

## Baseline

The Stage 1 baseline is **Nearest Neighbour**:

1. Start at the depot/station stop.
2. Among unvisited stops, choose the stop with the smallest travel time from the current stop.
3. Repeat until all stops are visited.
4. Compute total travel time for the generated path.

## Random Forest next-stop model

The ML task is binary next-stop prediction. Each supervised sample represents:

```text
current stop + candidate stop + route/stop features -> label
```

The label is `1` when the candidate is the historical next stop and `0` otherwise. Train/test splitting is performed by route ID, not by random rows, to avoid route leakage. A full route is predicted iteratively by choosing the candidate stop with the highest predicted probability at each step.

## Evaluation metrics

Stage 1 compares actual historical, Nearest Neighbour, and Random Forest sequences using:

- Total travel time.
- Percentage gap versus the actual historical sequence.
- Candidate-row next-stop prediction accuracy.
- Kendall Tau rank similarity.
- Spearman rank correlation.
- Edit distance.

## Google Colab

Open `notebooks/main_colab.ipynb` in Google Colab. The notebook installs requirements, locates the dataset, creates a small working subset, runs preprocessing, trains the Random Forest model, evaluates results, and displays Folium visualizations.
