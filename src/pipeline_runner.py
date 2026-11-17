import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
# pyrefly: ignore [missing-import]
import config
# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger
# pyrefly: ignore [missing-import]
from src.ingestion.download_accidents import download_accidents
# pyrefly: ignore [missing-import]
from src.ingestion.fetch_weather import fetch_weather_for_accidents
# pyrefly: ignore [missing-import]
from src.transformation.clean_and_validate import run_cleaning_pipeline
# pyrefly: ignore [missing-import]
from src.transformation.join_and_load import run_dw_load_pipeline

logger = get_logger("pipeline_runner")

def task_extract_accidents():
    """Task 1: Extract DfT Road Casualty accident data."""
    logger.info("========== TASK 1: EXTRACT ACCIDENTS ==========")
    raw_accidents_path = download_accidents()
    logger.info(f"Task 1 Complete. Output: {raw_accidents_path}")

def task_extract_weather():
    """Task 2: Extract Open-Meteo ERA5 hourly historical weather data."""
    logger.info("========== TASK 2: EXTRACT WEATHER ==========")
    raw_weather_path = fetch_weather_for_accidents()
    logger.info(f"Task 2 Complete. Output: {raw_weather_path}")

def task_validate_and_load():
    """Task 3: Clean, validate, spatio-temporal join, and load PostgreSQL/PostGIS database & mart."""
    logger.info("========== TASK 3: VALIDATE, TRANSFORM & LOAD ==========")
    staging_acc, staging_wea = run_cleaning_pipeline()
    run_dw_load_pipeline(staging_accidents_path=staging_acc, staging_weather_path=staging_wea)
    logger.info("Task 3 Complete. Database & Mart Loaded successfully.")

def run_full_pipeline():
    """Orchestrates the end-to-end Traffic Accident Analytics Pipeline."""
    start_time = time.time()
    logger.info("Starting Traffic Accident Severity Analytics Pipeline Execution...")
    
    task_extract_accidents()
    task_extract_weather()
    task_validate_and_load()

    elapsed = time.time() - start_time
    logger.info(f"========== PIPELINE EXECUTION SUCCESSFUL ({elapsed:.2f}s) ==========")

if __name__ == "__main__":
    run_full_pipeline()
