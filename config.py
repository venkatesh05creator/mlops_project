import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "traffic_analytics")

# SQLite Fallback Configuration (if PostgreSQL local service is absent)
USE_SQLITE_FALLBACK = os.getenv("USE_SQLITE_FALLBACK", "false").lower() == "true"
SQLITE_DB_PATH = BASE_DIR / "data" / "traffic_analytics.db"

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STAGING_DATA_DIR = DATA_DIR / "staging"
LOGS_DIR = DATA_DIR / "logs"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs & Subset Parameters
ACCIDENT_DATA_URL_2022 = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2022.csv"
ACCIDENT_DATA_URL_2023 = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2023.csv"

# Region Filter (e.g. Police Force 01 = Metropolitan Police / London area)
# Target Local Authorities / Police Forces
POLICE_FORCE_FILTER = int(os.getenv("POLICE_FORCE_FILTER", "1")) # 1 = Metropolitan Police

# Spatial Bounding Box for UK (Validation)
UK_BOUNDS = {
    "min_lat": 49.8,
    "max_lat": 60.9,
    "min_lon": -8.6,
    "max_lon": 1.8
}

# Weather API Endpoint (Open-Meteo Historical Weather)
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

def get_db_url():
    """Return database connection URL for SQLAlchemy."""
    if USE_SQLITE_FALLBACK:
        return f"sqlite:///{SQLITE_DB_PATH}"
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# MLOps Configuration
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR = BASE_DIR / "mlruns"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DB_PATH = BASE_DIR / "data" / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "traffic-severity-prediction")
MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "traffic_severity_classifier")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

