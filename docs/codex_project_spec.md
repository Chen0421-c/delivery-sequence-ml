# Codex Project Specification

## Role
Implement an MSc AI research project titled **Road Freight Delivery Sequence Optimization using Machine Learning**.

## Scope constraints
Do not turn this into a full multi-vehicle CVRPTW system. Focus on single-route, single-vehicle delivery stop sequencing using historical last-mile delivery data.

Avoid multi-vehicle routing, multi-depot routing, OR-Tools, XGBoost, deep reinforcement learning, Transformer/pointer networks, LLM route generation, and full production dashboards unless requested later.

## Dataset
Use the Amazon Last Mile Routing Research Challenge Dataset with route_data.json, actual_sequences.json, package_data.json, and travel_times.json.

Important: do not load the full travel_times.json into memory by default. Use memory-safe subset loading.

## Required modules
- src/data_loader.py
- src/preprocessing.py
- src/baseline.py
- src/ml_model.py
- src/evaluation.py
- src/visualization.py
- notebooks/main_colab.ipynb

## Tasks
1. Build memory-safe data loader with ijson.
2. Convert routes into stop tables and actual sequences.
3. Implement Actual Route comparison and the Stage 1 Nearest Neighbour baseline.
4. Build the Stage 1 Random Forest next-stop prediction model.
5. Generate predicted stop sequences iteratively.
6. Evaluate travel time, percentage gap, Kendall Tau, Spearman, edit distance, and prediction accuracy.
7. Create simple Folium route visualization and result charts.
8. Create a Colab notebook that runs the Stage 1 MVP on a default subset of 100 routes.

## Quality requirements
Use readable, well-commented Python. Avoid hard-coded local paths. Make paths configurable. Use Colab-friendly defaults. Fail gracefully when fields are missing.
