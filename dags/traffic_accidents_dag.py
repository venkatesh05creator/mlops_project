from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
from airflow import DAG
# pyrefly: ignore [missing-import]
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline_runner import (
    task_extract_accidents,
    task_extract_weather,
    task_validate_and_load
)

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    'traffic_accident_severity_analytics',
    default_args=default_args,
    description='ETL pipeline joining UK DfT road casualties with Open-Meteo ERA5 hourly weather',
    schedule_interval='0 2 * * *',  # Daily at 02:00 UTC
    catchup=False,
    tags=['mlops', 'postgis', 'weather', 'traffic']
) as dag:

    t1_extract_accidents = PythonOperator(
        task_id='extract_accidents',
        python_callable=task_extract_accidents
    )

    t2_extract_weather = PythonOperator(
        task_id='extract_weather',
        python_callable=task_extract_weather
    )

    t3_validate_and_load = PythonOperator(
        task_id='validate_and_load',
        python_callable=task_validate_and_load
    )

    t1_extract_accidents >> t2_extract_weather >> t3_validate_and_load
