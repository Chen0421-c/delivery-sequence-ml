# Road Freight Delivery Sequence Optimization using Machine Learning

This repository supports the MSc research project:

**Road Freight Delivery Sequence Optimization using Machine Learning**

## Project aim

The aim of this project is to develop a practical machine learning-based approach for improving single-route road freight delivery sequence optimization using historical last-mile delivery data.

## Research scope

This project focuses on single-route delivery stop sequencing, one delivery vehicle serving multiple stops from a depot/station, historical last-mile delivery data, baseline route sequencing methods, machine learning-based next-stop prediction, route performance evaluation, sequence similarity evaluation, and simple route visualization.

This project does not cover multi-vehicle fleet assignment, multi-depot routing, warehouse operations, parcel sorting, long-haul transportation, or full commercial logistics deployment.

## Planned methods

Dataset: Amazon Last Mile Routing Research Challenge Dataset.

Baseline methods:
1. Actual historical driver sequence
2. Nearest Neighbour sequencing
3. OR-Tools single-vehicle TSP baseline

Machine learning model:
- Supervised next-stop prediction
- Random Forest first, XGBoost optional

Evaluation metrics:
- Total travel time
- Percentage gap compared with baseline
- Next-stop prediction accuracy
- Kendall Tau correlation
- Spearman correlation
- Edit distance
