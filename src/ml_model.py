"""Machine learning utilities for next-stop prediction."""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def build_next_stop_training_samples(route_ids: List[str], route_data: Dict[str, Any], actual_sequences: Dict[str, Any], package_data: Dict[str, Any], travel_times: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    rows, labels = [], []
    for route_id in route_ids:
        seq_obj = actual_sequences[route_id]
        actual_seq = [sid for sid, order in sorted(seq_obj.items(), key=lambda x: x[1])] if isinstance(seq_obj, dict) else list(seq_obj)
        tt = travel_times[route_id]
        route_stops = route_data[route_id].get('stops', {})
        for pos in range(len(actual_seq)-1):
            current, true_next = actual_seq[pos], actual_seq[pos+1]
            current_info = route_stops.get(current, {})
            for cand in actual_seq[pos+1:]:
                cand_info = route_stops.get(cand, {})
                rows.append({
                    'travel_time_current_to_candidate': float(tt.get(current, {}).get(cand, 0.0)),
                    'current_lat': current_info.get('lat', 0.0) or 0.0,
                    'current_lng': current_info.get('lng', 0.0) or 0.0,
                    'candidate_lat': cand_info.get('lat', 0.0) or 0.0,
                    'candidate_lng': cand_info.get('lng', 0.0) or 0.0,
                    'same_zone': int(str(current_info.get('zone_id')) == str(cand_info.get('zone_id'))),
                    'route_stop_count': len(route_stops),
                    'current_position_ratio': pos / max(len(actual_seq)-1, 1),
                })
                labels.append(1 if cand == true_next else 0)
    return pd.DataFrame(rows), pd.Series(labels, name='is_next_stop')


def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3, class_weight='balanced', random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
