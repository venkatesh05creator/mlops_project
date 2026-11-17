import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ml.serve import app
from src.ml.pipeline import get_inference_pipeline

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_single_endpoint():
    payload = {
        "accident_hour": 18,
        "day_of_week": 5,
        "month": 10,
        "is_weekend": 0,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "speed_limit": 30,
        "road_class": "A",
        "road_surface": "Dry",
        "light_conditions": "Daylight",
        "weather_condition": "Mainly clear",
        "temperature_celsius": 14.0,
        "precipitation_mm": 0.0,
        "number_of_vehicles": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_severity"] in ["Slight", "Serious", "Fatal"]
    assert "probabilities" in data
    assert "confidence" in data
    assert "risk_level" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_batch_endpoint():
    payload = [
        {
            "accident_hour": 8,
            "day_of_week": 1,
            "month": 2,
            "speed_limit": 30,
            "road_surface": "Dry"
        },
        {
            "accident_hour": 23,
            "day_of_week": 6,
            "month": 12,
            "speed_limit": 70,
            "road_surface": "Frost or ice",
            "light_conditions": "Darkness - no lighting",
            "temperature_celsius": -2.0,
            "number_of_vehicles": 3
        }
    ]
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[1]["risk_level"] in ["Medium", "High"]

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "registry_name" in data
    assert "classes" in data
