# System Architecture & Data Flow Design

## Architectural Principles & Ownership

### 1. Single Responsibility & Concern Scoping
- **Configuration & Environment**: Centralized in [`config.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/config.py). `config.py` is the single owner of environment variables (`.env`), spatial bounding box definitions, database URLs, and directory paths.
- **Reference Lookups**: Encapsulated in [`src/transformation/lookups.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/src/transformation/lookups.py). Pure decodes mapping integer codes from official DfT STATS19 guides to text labels.
- **Run Manifests & Rejection Auditing**: [`src/utils/manifest.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/src/utils/manifest.py) owns execution audit records (extract timestamps, source URLs, file SHA256 hashes, row counts) and rejection logs (`runtime/logs/rejections.csv`).
- **Data Ingestion**:
  - [`src/ingestion/download_accidents.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/src/ingestion/download_accidents.py): Ingests DfT casualty CSVs into `data/raw/accidents_raw.csv`.
  - [`src/ingestion/fetch_weather.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/src/ingestion/fetch_weather.py): Ingests Open-Meteo ERA5 hourly historical weather into `data/raw/weather_raw.csv`.
- **Transformation & Validation**: [`src/transformation/clean_and_validate.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/src/transformation/clean_and_validate.py) enforces UK bounding box spatial constraints, deduplication, and lookup mapping before exporting parquet files to `data/staging/`.
- **Database Loading & PostGIS**: [`src/transformation/join_and_load.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/src/transformation/join_and_load.py) executes schema DDL [`sql/01_schema.sql`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/sql/01_schema.sql), spatio-temporal joins (grid cell + local hour), populates PostGIS `ST_Point` geometries, and creates analytical data marts.
- **Orchestration**:
  - Local CLI Runner: [`src/pipeline_runner.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/src/pipeline_runner.py) sequentially executes `extract_accidents` -> `extract_weather` -> `validate_and_load`.
  - Production Airflow DAG: [`dags/traffic_accidents_dag.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/dags/traffic_accidents_dag.py) wraps the 3 task callables.
- **Presentation**: [`app.py`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/app.py) provides 5 Streamlit analytical dashboard tabs querying `mart.accident_summary` and `analytics.fact_accident`.

---

## Strict 1-Way Data Flow Direction

```
[ DfT Casualty Data ] ---> download_accidents.py ---> data/raw/accidents_raw.csv
                                                             |
[ Open-Meteo ERA5 ]   ---> fetch_weather.py     ---> data/raw/weather_raw.csv
                                                             |
                                                   clean_and_validate.py
                                                             |
                                            data/staging/*.parquet + rejections.csv
                                                             |
                                                    join_and_load.py
                                                             |
                                        PostgreSQL 16 + PostGIS 3.6 (Port 5433)
                                           ├── staging.stg_*
                                           ├── analytics.fact_accident (PostGIS Point)
                                           └── mart.accident_summary
                                                             |
                                                    Streamlit Dashboard (app.py)
```

---

## State Ownership Map
| Storage Component | Path | Owner | Description |
| :--- | :--- | :--- | :--- |
| Database Engine | `runtime/pgdata` | PostgreSQL Process | Persistent spatial & dimensional database cluster |
| Raw Landed Files | `data/raw/` | Ingestion Modules | CSV landed files from DfT & Open-Meteo |
| Staging Parquet | `data/staging/` | Cleaning Module | Validated parquet datasets |
| Audit Trail & Logs | `runtime/logs/` | Manifest & Logger | JSON run manifests & `rejections.csv` |
