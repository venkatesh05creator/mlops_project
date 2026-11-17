# Traffic Accident Severity Analytics Report

UK DfT Road Casualty Statistics & Open-Meteo ERA5 Historical Weather Pipeline

---

## 1. Executive Summary

This report documents the design, architecture, implementation, and analytical findings of the **Traffic Accident Severity Analytics Pipeline**. The system ingests official road casualty collision data from the UK Department for Transport (DfT) and enriches it with high-resolution hourly ERA5 historical weather data from the Open-Meteo API.

The pipeline is implemented using Python, PostgreSQL 16 with PostGIS 3.6 spatial extensions, Apache Airflow 2.x, and Streamlit for interactive operational reporting. All data transformation, spatio-temporal grid matching, spatial indexing, and analytical mart aggregations are fully automated and verified.

---

## 2. System Architecture & Data Engineering Pipeline

### 2.1 Modular Architecture & Scoped Ownership

The codebase is structured following single-responsibility engineering principles:

- **`config.py`**: Centralized manager for global paths, environment configuration, database endpoints, and spatial bounding boxes.
- **`src/transformation/lookups.py`**: Encapsulates official DfT STATS19 lookup codebooks (collision severity, weather conditions, road types, lighting, road surface).
- **`src/utils/manifest.py`**: Generates cryptographic SHA-256 run manifests and records invalid rows into `runtime/logs/rejections.csv`.
- **`src/ingestion/`**: Downloads raw CSV datasets (`accidents_raw.csv` and `weather_raw.csv`) to the `data/raw/` landing zone.
- **`src/transformation/clean_and_validate.py`**: Applies spatial UK bounding box validation (`min_lat: 49.8, max_lat: 60.9, min_lon: -8.6, max_lon: 1.8`), date/time parsing, and deduplication.
- **`src/transformation/join_and_load.py`**: Executes spatio-temporal joins on `(grid_lat, grid_lon, match_time)`, creates PostGIS spatial points (`ST_SetSRID(ST_MakePoint(lon, lat), 4326)`), builds analytical star schemas (`analytics.fact_accident`), and populates operational data marts (`mart.accident_summary`).

```text
[ DfT Collision Data ] ---> Ingestion ---> data/raw/accidents_raw.csv
                                                |
[ Open-Meteo ERA5 ]    ---> Ingestion ---> data/raw/weather_raw.csv
                                                |
                                      Clean & Validate
                                                |
                                    data/staging/*.parquet
                                                |
                                       Spatio-Temporal Join
                                                |
                                  PostgreSQL 16 + PostGIS 3.6
                                    ├── staging.stg_collision
                                    ├── analytics.fact_accident (PostGIS Point)
                                    └── mart.accident_summary
                                                |
                                       Streamlit Dashboard
```

---

## 3. Data Quality, Validation & Audit Trail

### 3.1 Data Validation Rules

1. **Spatial Bounding Box Filter**: Coordinates outside valid UK bounds are rejected to prevent invalid geospatial queries.
2. **Temporal Parsing**: Collision timestamps are standardized to local ISO format (`YYYY-MM-DD HH:MM:SS`), handling multi-format inputs (`DD/MM/YYYY`, 4-digit HHMM strings).
3. **Deduplication**: Duplicate collision indices are dropped and logged to audit trails.
4. **Rejection Auditing**: Every rejected record is logged with timestamp and exact failure reason to `runtime/logs/rejections.csv`.

---

## 4. PostGIS Spatial Indexing & Star Schema

### 4.1 PostGIS Spatial Enablement

Collision points are converted into native PostGIS `GEOMETRY(Point, 4326)` columns. GiST spatial indexes (`idx_fact_accident_geom`) are applied to enable fast spatial proximity queries, hotspot bounding, and distance calculations.

### 4.2 Data Mart Design

The data mart table `mart.accident_summary` pre-aggregates accident counts, vehicle involvement, and casualty totals grouped by year, month, hour, day of week, severity label, road surface, and weather condition for sub-second Streamlit dashboard response times.

---

## 5. Dashboard & Analytical Findings

The Streamlit operational dashboard (`app.py`) provides 5 core analytical views:

1. **Severity Distribution**: Proportional breakdown of Fatal, Serious, and Slight collisions.
2. **Temporal Heatmap**: Collision volume matrix mapped across 24 hours of the day vs days of the week.
3. **Weather & Road Surface Impact**: Comparative analysis of collisions occurring during rainfall/snow vs fine conditions, dry vs wet/damp roads.
4. **PostGIS Spatial Hotspot Map**: Interactive PyDeck geospatial map rendering collision density across coordinates.
5. **Vehicle & Casualty Analysis**: Frequency distribution of casualties and vehicles involved per incident.

---

## 6. Verification & Reproducibility Guide

To run and verify the entire system locally on Windows:

```bash
# 1. Setup portable PostgreSQL 16 + PostGIS 3.6 server (port 5433)
.venv\Scripts\python.exe scripts\setup_postgres_runtime.py

# 2. Run automated unit tests
.venv\Scripts\python.exe -m pytest

# 3. Execute full ETL pipeline
.venv\Scripts\python.exe -m src.pipeline_runner

# 4. Launch Streamlit Analytics Dashboard
.venv\Scripts\streamlit.exe run app.py --server.port 8501
```
