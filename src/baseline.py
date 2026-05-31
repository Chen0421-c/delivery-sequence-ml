"""Baseline route sequencing methods."""
from typing import Dict, List


def calculate_sequence_travel_time(sequence: List[str], travel_time_matrix: Dict[str, Dict[str, float]]) -> float:
    return sum(float(travel_time_matrix.get(a, {}).get(b, 0.0)) for a, b in zip(sequence[:-1], sequence[1:]))


def nearest_neighbour_sequence(stop_ids: List[str], depot_id: str, travel_time_matrix: Dict[str, Dict[str, float]]) -> List[str]:
    unvisited = set(stop_ids)
    if depot_id in unvisited:
        unvisited.remove(depot_id)
    sequence = [depot_id]
    current = depot_id
    while unvisited:
        next_stop = min(unvisited, key=lambda s: float(travel_time_matrix.get(current, {}).get(s, float("inf"))))
        sequence.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop
    return sequence


def ortools_tsp_sequence(stop_ids: List[str], depot_id: str, travel_time_matrix: Dict[str, Dict[str, float]], time_limit_seconds: int = 5) -> List[str]:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return nearest_neighbour_sequence(stop_ids, depot_id, travel_time_matrix)
    stops = [depot_id] + [s for s in stop_ids if s != depot_id]
    try:
        manager = pywrapcp.RoutingIndexManager(len(stops), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        def cb(from_index, to_index):
            a = stops[manager.IndexToNode(from_index)]
            b = stops[manager.IndexToNode(to_index)]
            return int(float(travel_time_matrix.get(a, {}).get(b, 10**6)))
        transit_callback_index = routing.RegisterTransitCallback(cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.time_limit.FromSeconds(time_limit_seconds)
        solution = routing.SolveWithParameters(search_params)
        if solution is None:
            return nearest_neighbour_sequence(stop_ids, depot_id, travel_time_matrix)
        sequence, index = [], routing.Start(0)
        while not routing.IsEnd(index):
            sequence.append(stops[manager.IndexToNode(index)])
            index = solution.Value(routing.NextVar(index))
        return sequence
    except Exception:
        return nearest_neighbour_sequence(stop_ids, depot_id, travel_time_matrix)
