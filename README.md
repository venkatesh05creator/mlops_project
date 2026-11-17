# Traffic Accident Severity Analytics — Data Engineering Pipeline & Dashboard

Part 1 of the Data Engineering & MLOps Course Project. Implements an end-to-end data pipeline and interactive visual dashboard analyzing UK DfT Road Casualty accident severity joined with Open-Meteo ERA5 hourly historical weather.

---

## 🌟 Architecture Overview

```text
Source Data (UK DfT STATS19 CSV + Open-Meteo ERA5 API)
   │
   ▼
Ingestion Layer (download_accidents.py & fetch_weather.py)
   │
   ▼
Raw Landing Zone (data/raw/)
   │
   ▼
Airflow / CLI Pipeline Runner (dags/traffic_accidents_dag.py & src/pipeline_runner.py)
   │
   ▼
Transformation & Cleaning (clean_and_validate.py — UK Bounding Box & Rejection Auditing)
   │
   ▼
Database DW & Mart (PostgreSQL 16 + PostGIS 3.6 — Star Schema)
   │
   ▼
Streamlit Visual Analytics Dashboard (app.py — 5 Analytical Views)
```

---

## ⚙️ Environment Variables & Configuration

`.env` configuration:

```ini
DB_HOST=localhost
DB_PORT=5433
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=traffic
POLICE_FORCE_FILTER=1
USE_SQLITE_FALLBACK=false
```

---

## 🚀 Quickstart & Setup Guide

### 1. Install Dependencies (Includes Apache Airflow & Pipeline Core)

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Set Up Portable PostgreSQL 16 + PostGIS 3.6 (Port 5433)

```powershell
.venv\Scripts\python.exe scripts\setup_postgres_runtime.py
```

### 3. Run Unit Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

### 4. Run End-to-End Pipeline

```powershell
.venv\Scripts\python.exe -m src.pipeline_runner
```

### 5. Launch Streamlit Analytics Dashboard

```powershell
.venv\Scripts\streamlit.exe run app.py --server.port 8501
```

Open your browser to `http://localhost:8501`.

---

## 📚 Documentation

- [`docs/ARCHITECTURE.md`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/docs/ARCHITECTURE.md): System architecture, modularity, state ownership, and data flow.
- [`docs/DATA_DICTIONARY.md`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/docs/DATA_DICTIONARY.md): Staging, analytics, and mart schema definitions.
- [`docs/REPORT.md`](file:///c:/Users/Venkatesh/OneDrive/Desktop/MLOPS_ASSIGNMENT/docs/REPORT.md): Comprehensive project report and analytical insights.
