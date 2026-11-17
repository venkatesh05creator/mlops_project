import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils.logger import get_logger

logger = get_logger("ml_monitoring")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float:
    """
    Calculates the Population Stability Index (PSI) between a baseline distribution
    (expected) and production inference data (actual).
    
    Interpretation:
    - PSI < 0.10: No significant shift (distribution is stable)
    - 0.10 <= PSI < 0.25: Moderate shift (monitor closely)
    - PSI >= 0.25: Significant shift (trigger retraining)
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] -= 1e-5
    bucket_bounds[-1] += 1e-5

    exp_counts, _ = np.histogram(expected, bins=bucket_bounds)
    act_counts, _ = np.histogram(actual, bins=bucket_bounds)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Apply epsilon smoothing to avoid log(0) or division by zero
    exp_pct = np.clip(exp_pct, epsilon, 1.0)
    act_pct = np.clip(act_pct, epsilon, 1.0)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


def monitor_class_drift(baseline_proportions: Dict[str, float], prod_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Monitors class distribution drift between training baseline and production predictions.
    Uses Jensen-Shannon (JS) divergence and proportional divergence.
    """
    classes = ["Slight", "Serious", "Fatal"]
    total_prod = max(sum(prod_counts.values()), 1)
    prod_proportions = {c: prod_counts.get(c, 0) / total_prod for c in classes}

    p = np.array([baseline_proportions.get(c, 0.0) for c in classes])
    q = np.array([prod_proportions[c] for c in classes])

    js_divergence = float(jensenshannon(p, q))
    drift_detected = js_divergence > 0.15

    return {
        "metric": "Jensen-Shannon Divergence",
        "js_divergence": round(js_divergence, 4),
        "threshold": 0.15,
        "drift_detected": drift_detected,
        "baseline_proportions": baseline_proportions,
        "production_proportions": {k: round(v, 4) for k, v in prod_proportions.items()}
    }


def monitor_spatial_location_drift(
    baseline_lats: np.ndarray,
    baseline_lons: np.ndarray,
    prod_lats: np.ndarray,
    prod_lons: np.ndarray,
    max_centroid_shift_km: float = 5.0
) -> Dict[str, Any]:
    """
    Monitors spatial drift in accident locations (e.g. shifts from Central London
    to outer borough expressways).
    """
    base_center_lat, base_center_lon = np.mean(baseline_lats), np.mean(baseline_lons)
    prod_center_lat, prod_center_lon = np.mean(prod_lats), np.mean(prod_lons)

    # Approximate Euclidean km distance in UK latitude
    dlat = (prod_center_lat - base_center_lat) * 111.0
    dlon = (prod_center_lon - base_center_lon) * 111.0 * np.cos(np.radians(base_center_lat))
    centroid_shift_km = float(np.sqrt(dlat**2 + dlon**2))

    lat_psi = calculate_psi(baseline_lats, prod_lats)
    lon_psi = calculate_psi(baseline_lons, prod_lons)

    drift_detected = (centroid_shift_km > max_centroid_shift_km) or (lat_psi > 0.20 or lon_psi > 0.20)

    return {
        "centroid_shift_km": round(centroid_shift_km, 2),
        "threshold_km": max_centroid_shift_km,
        "latitude_psi": round(lat_psi, 4),
        "longitude_psi": round(lon_psi, 4),
        "drift_detected": drift_detected,
        "alert_level": "WARNING" if drift_detected else "HEALTHY"
    }


def evaluate_high_severity_fnr(y_ground_truth: np.ndarray, y_predicted: np.ndarray, max_fnr_threshold: float = 0.55) -> Dict[str, Any]:
    """
    Monitors False Negative Rate on High-Severity (Serious + Fatal) accidents.
    Prioritizes identifying missed critical accidents.
    """
    # High severity is class 1 (Serious) or class 2 (Fatal)
    actual_severe = (y_ground_truth >= 1)
    total_actual_severe = int(np.sum(actual_severe))

    if total_actual_severe == 0:
        return {"error": "No ground truth severe cases in evaluation batch"}

    # False negative: actual severe was predicted as Slight (0)
    false_negatives = int(np.sum((y_ground_truth >= 1) & (y_predicted == 0)))
    true_positives = total_actual_severe - false_negatives

    fnr = false_negatives / total_actual_severe
    recall = true_positives / total_actual_severe
    alert_triggered = fnr > max_fnr_threshold

    return {
        "total_actual_severe": total_actual_severe,
        "false_negatives_slight_predicted": false_negatives,
        "true_positives_detected": true_positives,
        "high_severity_recall": round(float(recall), 4),
        "high_severity_fnr": round(float(fnr), 4),
        "max_allowable_fnr": max_fnr_threshold,
        "alert_triggered": alert_triggered,
        "recommendation": "Retrain model immediately with adjusted class weights" if alert_triggered else "Model within acceptable safety tolerance"
    }


if __name__ == "__main__":
    print("Testing ML Monitoring Module...")
    # Baseline simulation
    base_props = {"Slight": 0.8385, "Serious": 0.1572, "Fatal": 0.0043}
    prod_counts_stable = {"Slight": 830, "Serious": 165, "Fatal": 5}
    prod_counts_drifted = {"Slight": 600, "Serious": 350, "Fatal": 50}

    print("\n1. Stable Class Check:", monitor_class_drift(base_props, prod_counts_stable))
    print("\n2. Drifted Class Check:", monitor_class_drift(base_props, prod_counts_drifted))

    # FNR Check
    y_true = np.array([0, 0, 1, 1, 2, 0, 1, 2, 0, 0])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    print("\n3. High-Severity FNR Check:", evaluate_high_severity_fnr(y_true, y_pred))
