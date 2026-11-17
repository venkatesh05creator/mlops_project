# Technical Project Report: Traffic Accident Severity Analytics

**Course Project — Part 1: Data Engineering & MLOps Infrastructure**  
**Author**: Data Engineering & MLOps Team  
**Date**: September 2026  

---

## Executive Summary

Road traffic collisions represent a critical public safety challenge worldwide. Analyzing spatio-temporal collision patterns alongside historical meteorological conditions enables urban planners, transportation authorities, and emergency response teams to deploy targeted interventions.

This project delivers **Part 1 of the Data Engineering & MLOps Course Project: "Traffic Accident Severity Analytics"** — a robust, automated end-to-end data processing pipeline and interactive visual dashboard. The system ingests official UK Department for Transport (DfT) STATS19 collision records and enriches them with historical hourly weather metrics (temperature, precipitation, wind speed, humidity) fetched dynamically from the Open-Meteo REST API.

Key achievements of this Part 1 implementation:
1. Ingested **23,327** validated collision records for Greater London (Metropolitan Police region) alongside **720** historical hourly meteorological data points.
2. Built an automated 3-task **Apache Airflow DAG** (`extract_accidents` $\rightarrow$ `extract_weather` $\rightarrow$ `validate_and_load`).
3. Designed and populated a PostgreSQL / PostGIS **Star Schema Data Warehouse** containing 5 dimension tables (`dim_date`, `dim_location`, `dim_road`, `dim_weather`, `dim_severity`) and a centralized `fact_accidents` table.
4. Created pre-aggregated analytical marts (`mart.monthly_hourly_accident_summary`).
5. Implemented a 5-view **Streamlit Analytical Dashboard** with PyDeck 3D geospatial hotspot visualizations.
6. Established a clean integration interface for **Part 2 (MLOps & Model Training)**.

---

## 1. Introduction & Motivation

Understanding traffic accident severity requires integrating disparate datasets. Primary collision records capture incident attributes (location, road surface, speed limit, casualty count), while environmental factors (rain, fog, wind, ambient temperature) heavily influence road traction and driver visibility.

### Objectives:
- **Completeness & Demonstrability**: Build a fully functional, production-ready pipeline prioritizing end-to-end data integrity over massive unmanageable file sizes.
- **Architectural Separation**: Implement strict layer boundaries (Source $\rightarrow$ Raw $\rightarrow$ Staging $\rightarrow$ DW Star Schema $\rightarrow$ Analytical Mart $\rightarrow$ Streamlit Dashboard).
- **Auditability**: Log extraction timestamps, source URLs, record counts, file SHA256 hashes, and exact rejection reasons for invalid records.

---

## 2. Dataset Selection & Ingestion Strategy

### 2.1 Primary Data Source: UK STATS19 Road Collision Data
The UK Department for Transport publishes validated personal injury collision microdata. For scope management and low latency, we selected the **2022 STATS19 Collision Dataset** filtered to **Police Force 01 (Metropolitan Police / Greater London)**.

- **Total UK Dataset Size**: 106,004 collisions (19.18 MB CSV).
- **Scoped Region Subset**: 23,327 collisions in Greater London.
- **Audit Hash (SHA256)**: `dbc8c3282bdad1876472cddc27c38269f19ad90ca77bdf75a27e17b394dcbd04`.

### 2.2 Secondary Data Source: Open-Meteo Historical Weather API
Open-Meteo provides high-resolution historical weather reanalysis data without requiring API keys. To respect rate limits and minimize redundant network requests, we implemented spatial grid rounding ($0.1^\circ \approx 10\text{ km}$ resolution).

- **Weather Parameters Ingested**: Temperature at 2m ($^\circ\text{C}$), Hourly Precipitation ($\text{mm}$), Wind Speed at 10m ($\text{km/h}$), Relative Humidity ($\%$), and WMO Weather Condition Codes.
- **Hourly Records Fetched**: 720 weather hours with **0** API errors.

---

## 3. Data Cleaning, Quality Assurance & Validation

Data quality checks run during Phase 3 (`src/transformation/clean_and_validate.py`):

1. **Spatial Bounding Box Validation**: Coordinates must fall within the UK geographical envelope:
   $$49.8 \le \text{Latitude} \le 60.9, \quad -8.6 \le \text{Longitude} \le 1.8$$
   Records outside these bounds are automatically rejected and appended to `data/logs/error_records.log`.
2. **Date & Timestamp Normalization**: Standardized string dates (`DD/MM/YYYY`) and time fields into ISO datetime objects (`YYYY-MM-DD HH:MM:SS`).
3. **Categorical Standardizations**:
   - Severity Codes: `1` $\rightarrow$ "Fatal", `2` $\rightarrow$ "Serious", `3` $\rightarrow$ "Slight".
   - Road Surface Codes: `1` $\rightarrow$ "Dry", `2` $\rightarrow$ "Wet or damp", `3` $\rightarrow$ "Snow", `4` $\rightarrow$ "Frost or ice".
4. **Deduplication**: Deduplicated on unique collision identifier (`accident_index`).

---

## 4. Database Modeling: PostgreSQL / PostGIS Star Schema

We implemented a classic Kimball dimensional model supporting multi-dimensional OLAP analytics.

```
       +-------------------+       +--------------------+
       |   dw.dim_date     |       |   dw.dim_location  |
       +-------------------+       +--------------------+
       | PK date_key       |<---+  | PK location_key    |<---+
       +-------------------+    |  +--------------------+    |
                                |                            |
       +-------------------+    |  +--------------------+    |
       |   dw.dim_road     |    |  |  dw.fact_accidents |    |
       +-------------------+    |  +--------------------+    |
       | PK road_key       |<---+--| PK accident_index  |    |
       +-------------------+    |  | FK date_key        |----+
                                |  | FK location_key    |
       +-------------------+    |  | FK road_key        |
       |  dw.dim_weather   |    |  | FK weather_key     |----+
       +-------------------+    |  | FK severity_key    |    |
       | PK weather_key    |<---+  +--------------------+    |
       +-------------------+                                 |
                                   +--------------------+    |
                                   |  dw.dim_severity   |    |
                                   +--------------------+    |
                                   | PK severity_key    |<---+
                                   +--------------------+
```

### Table Population Summary:
- `dw.dim_date`: 365 date dimension rows.
- `dw.dim_location`: Distinct spatial locations with PostGIS `GEOMETRY(Point, 4326)`.
- `dw.dim_road`: Distinct road classifications, speed limits, and surface states.
- `dw.dim_weather`: Distinct hourly weather observation vectors.
- `dw.dim_severity`: Severity classifications and vehicle/casualty metrics.
- `dw.fact_accidents`: **23,327** fact records.
- `mart.monthly_hourly_accident_summary`: **5,104** aggregated summary records.

---

## 5. Pipeline Orchestration: Apache Airflow

The pipeline lifecycle is orchestrated via Apache Airflow (`dags/accident_pipeline_dag.py`).

```
[extract_accidents] ──► [extract_weather] ──► [validate_and_load]
```

- **Task 1: `extract_accidents`**: Downloads STATS19 CSV, applies regional filter, and records audit logs.
- **Task 2: `extract_weather`**: Queries Open-Meteo API for historical weather matching collision dates/locations.
- **Task 3: `validate_and_load`**: Cleans data, executes spatio-temporal joins, populates Star Schema DW, and updates analytical marts.

---

## 6. Visualization: Streamlit Analytical Dashboard

The Streamlit dashboard (`app.py`) provides 5 interactive analytical views:

1. **View 1 — Severity Distribution**: Visualizes the distribution of Slight (~86%), Serious (~13%), and Fatal (~1%) collisions.
2. **View 2 — Temporal Patterns**: Line charts of hourly accident trends showing peak volume during morning (08:00–09:00) and evening (17:00–18:30) rush hours, plus day-of-week matrix heatmaps.
3. **View 3 — Weather & Road Impact**: Comparative bar charts illustrating collision rates on dry vs wet/damp roads and across clear vs rainy weather conditions.
4. **View 4 — Geospatial Hotspot Map**: 3D PyDeck scatter plot visualizing accident coordinates across London boroughs.
5. **View 5 — Vehicle & Casualty Analysis**: Distributions showing multi-vehicle pileups and casualty counts per incident.

---

## 7. Part 2 Integration Roadmap (MLOps & Prediction)

Although Machine Learning logic is strictly excluded from Part 1, the pipeline provides a clean integration boundary for Part 2:

1. **Feature Store Integration**: Export `dw.fact_accidents` joined with `dw.dim_weather` and `dw.dim_road` as training feature sets (features: speed limit, weather condition, temperature, hour of day, day of week, road surface).
2. **Model Training & Registry**: Train an XGBoost / LightGBM Multi-Class Classifier to predict accident severity (`Slight`, `Serious`, `Fatal`) and track model metrics using MLflow.
3. **Model Serving & Monitoring**: Package the trained model into a REST API endpoint (FastAPI) and integrate real-time inference into the Streamlit dashboard.

---

## 8. Conclusion

Part 1 of the "Traffic Accident Severity Analytics" project has been successfully built and verified. The system demonstrates a complete, reliable data engineering workflow — from raw CSV download and external weather API integration to spatial validation, Star Schema storage, and interactive Streamlit analytics. All code is clean, modular, and fully documented for turn-key execution.
