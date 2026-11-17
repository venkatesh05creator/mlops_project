"""
Apache Airflow DAG: Traffic Accident Severity Analytics Pipeline
Orchestrates: extract_accidents -> extract_weather -> validate_and_load
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import custom ingestion and transformation modules
from src.ingestion.download_accidents import fetch_and_filter_accidents
from src.ingestion.fetch_weather import fetch_weather_for_accidents
from src.transformation.clean_and_validate import run_cleaning_pipeline
from src.transformation.join_and_load import run_dw_load_pipeline

default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'traffic_accident_severity_pipeline',
    default_args=default_args,
    description='End-to-end Data Engineering pipeline for UK Traffic Accidents & Weather',
    schedule_interval='@monthly',
    catchup=False
)

def task_extract_accidents():
    """Task 1: Pull raw accident CSV & apply regional filter."""
    fetch_and_filter_accidents()

def task_extract_weather():
    """Task 2: Call Open-Meteo API for historical weather by lat/lon & date."""
    fetch_weather_for_accidents()

def task_validate_and_load():
    """Task 3: Clean, spatial validate, join with weather, and load Star Schema & Mart."""
    cleaned_accidents_path, cleaned_weather_path = run_cleaning_pipeline()
    run_dw_load_pipeline(cleaned_accidents_path, cleaned_weather_path)

# Define DAG Operator Tasks
t1_extract_accidents = PythonOperator(
    task_id='extract_accidents',
    python_callable=task_extract_accidents,
    dag=dag,
)

t2_extract_weather = PythonOperator(
    task_id='extract_weather',
    python_callable=task_extract_weather,
    dag=dag,
)

t3_validate_and_load = PythonOperator(
    task_id='validate_and_load',
    python_callable=task_validate_and_load,
    dag=dag,
)

# Task Dependency Pipeline Flow
t1_extract_accidents >> t2_extract_weather >> t3_validate_and_load
