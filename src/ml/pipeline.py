import sys
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils.logger import get_logger
from src.ml.features import (
    TrafficFeatureExtractor,
    SEVERITY_LABEL_TO_INT,
    SEVERITY_INT_TO_LABEL,
)

logger = get_logger("ml_inference_pipeline")


class SeverityInferencePipeline:
    """
    Production inference wrapper that loads the trained end-to-end pipeline
    (FeatureExtractor -> Preprocessor -> Classifier) and provides high-level
    prediction APIs for raw traffic collision features.
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        if model_path is None:
            model_path = config.MODELS_DIR / "production_pipeline.joblib"
        self.model_path = Path(model_path)
        self.pipeline = None
        self.metadata = None
        self._load_pipeline()

    def _load_pipeline(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Production model pipeline not found at {self.model_path}. "
                f"Please run 'python src/ml/train.py' first."
            )
        logger.info(f"Loading trained inference pipeline from: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

        metadata_path = config.MODELS_DIR / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

    def predict(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw accident features as a dictionary and returns predicted severity,
        probabilities, and risk category.
        """
        df_input = pd.DataFrame([raw_input])
        return self.predict_batch(df_input)[0]

    def predict_batch(self, df_input: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Predicts severity for a batch of raw accident records.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not loaded.")

        # Ensure required columns exist with sensible defaults if missing
        df = df_input.copy()
        if "accident_hour" not in df.columns:
            df["accident_hour"] = 12
        if "day_of_week" not in df.columns:
            df["day_of_week"] = 3
        if "month" not in df.columns:
            df["month"] = 6
        if "is_weekend" not in df.columns:
            df["is_weekend"] = 0
        if "latitude" not in df.columns:
            df["latitude"] = 51.5074
        if "longitude" not in df.columns:
            df["longitude"] = -0.1278
        if "speed_limit" not in df.columns:
            df["speed_limit"] = 30
        if "road_class" not in df.columns:
            df["road_class"] = "A"
        if "road_surface" not in df.columns:
            df["road_surface"] = "Dry"
        if "light_conditions" not in df.columns:
            df["light_conditions"] = "Daylight"
        if "weather_condition" not in df.columns:
            df["weather_condition"] = "Mainly clear"
        if "temperature_celsius" not in df.columns:
            df["temperature_celsius"] = 10.0
        if "precipitation_mm" not in df.columns:
            df["precipitation_mm"] = 0.0
        if "number_of_vehicles" not in df.columns:
            df["number_of_vehicles"] = 1

        pred_classes = self.pipeline.predict(df)
        pred_probas = self.pipeline.predict_proba(df)

        results = []
        for cls_id, proba in zip(pred_classes, pred_probas):
            cls_name = SEVERITY_INT_TO_LABEL.get(int(cls_id), "Unknown")
            
            # Map probabilities to class names
            proba_dict = {
                "Slight": round(float(proba[0]), 4),
                "Serious": round(float(proba[1]), 4),
                "Fatal": round(float(proba[2]), 4)
            }
            
            # Determine priority risk classification
            high_sev_score = proba_dict["Serious"] + proba_dict["Fatal"]
            if proba_dict["Fatal"] >= 0.15 or high_sev_score >= 0.50:
                risk_level = "High"
            elif high_sev_score >= 0.25:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            results.append({
                "predicted_severity": cls_name,
                "predicted_class_id": int(cls_id),
                "confidence": round(float(np.max(proba)), 4),
                "probabilities": proba_dict,
                "high_severity_combined_probability": round(float(high_sev_score), 4),
                "risk_level": risk_level
            })

        return results


# Global singleton instance for serving
_inference_instance = None

def get_inference_pipeline() -> SeverityInferencePipeline:
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = SeverityInferencePipeline()
    return _inference_instance


if __name__ == "__main__":
    pipeline = get_inference_pipeline()
    sample_raw = {
        "accident_hour": 23,
        "day_of_week": 6,
        "month": 11,
        "is_weekend": 1,
        "latitude": 51.52,
        "longitude": -0.10,
        "speed_limit": 60,
        "road_class": "A",
        "road_surface": "Frost or ice",
        "light_conditions": "Darkness - no lighting",
        "weather_condition": "Light drizzle",
        "temperature_celsius": -1.0,
        "precipitation_mm": 0.8,
        "number_of_vehicles": 2
    }
    pred = pipeline.predict(sample_raw)
    print("Sample Prediction Output:")
    print(json.dumps(pred, indent=2))
