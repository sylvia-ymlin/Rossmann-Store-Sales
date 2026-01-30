import json
import os
import pandas as pd
from datetime import datetime
from src.core import setup_logger

logger = setup_logger(__name__)

class ExperimentTracker:
    def __init__(self, log_path='logs/experiments.json'):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
    def log_experiment(self, name, params, metrics):
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "params": params,
            "metrics": metrics
        }
        experiments = []
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                try:
                    experiments = json.load(f)
                except json.JSONDecodeError:
                    experiments = []
        experiments.append(experiment)
        with open(self.log_path, 'w') as f:
            json.dump(experiments, f, indent=4)

class DriftDetector:
    def __init__(self, baseline_stats=None):
        self.baseline_stats = baseline_stats
        
    def calculate_stats(self, df, features):
        return df[features].agg(['mean', 'std']).to_dict()
        
    def check_drift(self, df, features, threshold=0.2):
        if self.baseline_stats is None:
            self.baseline_stats = self.calculate_stats(df, features)
            return False, {}
        current_stats = self.calculate_stats(df, features)
        drifts = {}
        drift_detected = False
        for feature in features:
            base_mean = self.baseline_stats[feature]['mean']
            curr_mean = current_stats[feature]['mean']
            if base_mean != 0:
                change = abs(curr_mean - base_mean) / abs(base_mean)
                if change > threshold:
                    drifts[feature] = change
                    drift_detected = True
        return drift_detected, drifts
