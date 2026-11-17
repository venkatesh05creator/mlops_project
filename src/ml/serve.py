import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils.logger import get_logger
from src.ml.pipeline import get_inference_pipeline, SeverityInferencePipeline

logger = get_logger("ml_api_service")

# Initialize FastAPI App
app = FastAPI(
    title="Traffic Accident Severity Prediction Service",
    description="Production MLOps inference microservice for predicting traffic collision severity in real-time.",
    version="2.0.0"
)

# Enable CORS for Streamlit and external frontend consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class AccidentFeaturesInput(BaseModel):
    accident_hour: int = Field(default=14, ge=0, le=23, description="Hour of collision (0-23)")
    day_of_week: int = Field(default=3, ge=1, le=7, description="Day of week (1=Monday, 7=Sunday)")
    month: int = Field(default=6, ge=1, le=12, description="Month of year (1-12)")
    is_weekend: Optional[int] = Field(default=0, ge=0, le=1, description="1 if Saturday/Sunday else 0")
    latitude: float = Field(default=51.5074, description="Spatial latitude coordinate (e.g. 51.5074)")
    longitude: float = Field(default=-0.1278, description="Spatial longitude coordinate (e.g. -0.1278)")
    speed_limit: int = Field(default=30, ge=10, le=70, description="Posted road speed limit (mph)")
    road_class: str = Field(default="A", description="Road classification (A, B, Motorway, Unclassified)")
    road_surface: str = Field(default="Dry", description="Road surface condition (Dry, Wet or damp, Frost or ice, Snow, etc.)")
    light_conditions: str = Field(default="Daylight", description="Lighting conditions (Daylight, Darkness - lights lit, Darkness - no lighting, etc.)")
    weather_condition: str = Field(default="Mainly clear", description="Weather condition (Clear sky, Mainly clear, Light drizzle, Slight rain, etc.)")
    temperature_celsius: float = Field(default=12.0, description="Ambient air temperature in Celsius")
    precipitation_mm: float = Field(default=0.0, ge=0.0, description="Hourly precipitation in mm")
    number_of_vehicles: int = Field(default=1, ge=1, le=20, description="Number of vehicles involved in incident")

    model_config = {
        "json_schema_extra": {
            "example": {
                "accident_hour": 22,
                "day_of_week": 6,
                "month": 11,
                "is_weekend": 1,
                "latitude": 51.515,
                "longitude": -0.141,
                "speed_limit": 50,
                "road_class": "A",
                "road_surface": "Wet or damp",
                "light_conditions": "Darkness - lights lit",
                "weather_condition": "Light drizzle",
                "temperature_celsius": 3.5,
                "precipitation_mm": 1.2,
                "number_of_vehicles": 2
            }
        }
    }


class SeverityPredictionResponse(BaseModel):
    predicted_severity: str = Field(..., description="Predicted severity class (Slight, Serious, Fatal)")
    predicted_class_id: int = Field(..., description="Numeric class code (0=Slight, 1=Serious, 2=Fatal)")
    confidence: float = Field(..., description="Maximum predicted class confidence score (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution for Slight, Serious, Fatal")
    high_severity_combined_probability: float = Field(..., description="Combined probability of Serious + Fatal")
    risk_level: str = Field(..., description="Safety priority tier (Low, Medium, High)")
    inference_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
def root():
    return {
        "service": "Traffic Accident Severity Prediction API",
        "version": "2.0.0",
        "status": "online",
        "model_registry_name": config.MODEL_REGISTRY_NAME,
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
def healthcheck():
    try:
        pipeline = get_inference_pipeline()
        return {
            "status": "healthy",
            "model_loaded": pipeline.pipeline is not None,
            "model_metadata": pipeline.metadata,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Healthcheck failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/predict", response_model=SeverityPredictionResponse, tags=["Inference"])
def predict_severity(payload: AccidentFeaturesInput):
    """
    Predict severity class and probabilities for a single accident scenario.
    """
    try:
        pipeline = get_inference_pipeline()
        features = payload.model_dump()
        result = pipeline.predict(features)
        return result
    except Exception as e:
        logger.error(f"Inference error during /predict: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict_batch", response_model=List[SeverityPredictionResponse], tags=["Inference"])
def predict_severity_batch(payload: List[AccidentFeaturesInput]):
    """
    Predict severity class and probabilities for a batch of accident records.
    """
    try:
        pipeline = get_inference_pipeline()
        import pandas as pd
        records = [p.model_dump() for p in payload]
        df_batch = pd.DataFrame(records)
        results = pipeline.predict_batch(df_batch)
        return results
    except Exception as e:
        logger.error(f"Inference error during /predict_batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/model-info", tags=["Metadata"])
def get_model_info():
    """
    Returns registered model metadata, training run metrics, and class definitions.
    """
    pipeline = get_inference_pipeline()
    return {
        "registry_name": config.MODEL_REGISTRY_NAME,
        "metadata": pipeline.metadata,
        "classes": {
            0: "Slight",
            1: "Serious",
            2: "Fatal"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.ml.serve:app", host=config.API_HOST, port=config.API_PORT, reload=False)
