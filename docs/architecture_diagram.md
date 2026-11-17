# System Architecture Diagram — Traffic Accident Severity Analytics

## High-Level End-to-End Pipeline Architecture

```mermaid
flowchart TD
    subgraph Layer1["Layer 1: Data Sources"]
        A1["UK Road Safety STATS19 CSV<n(Department for Transport)"]
        A2["Open-Meteo Historical Weather API<n(Hourly Temp, Precip, Wind)"]
        A3["OpenStreetMap GIS Context<n(Road Network Geometry - Optional)"]
    end

    subgraph Layer2["Layer 2: Ingestion Engine"]
        B1["download_accidents.py<n(CSV Fetch & Regional Filtering)"]
        B2["fetch_weather.py<n(Spatio-temporal API Batch Fetcher)"]
        B3["Ingestion Audit Logger<n(Timestamps, Row Counts, SHA256)"]
    end

    subgraph Layer3["Layer 3: Raw Landing Zone"]
        C1[("data/raw/accidents_raw.csv")]
        C2[("data/raw/weather_raw.csv")]
    end

    subgraph Layer4["Layer 4: Orchestration & ETL Engine"]
        D1["Apache Airflow / Pipeline Runner<n(extract_accidents -> extract_weather -> validate_and_load)"]
        D2["clean_and_validate.py<n(Date Standardization, UK Bounding Box, Deduplication)"]
        D3["error_records.log<n(Audit Rejection Logs with Reasons)"]
        D4["join_and_load.py<n(Spatio-Temporal Match & Star Schema Population)"]
    end

    subgraph Layer5["Layer 5: Staging & Storage"]
        E1[("data/staging/accidents_clean.csv")]
        E2[("data/staging/weather_clean.csv")]
    end

    subgraph Layer6["Layer 6: PostgreSQL / PostGIS Data Warehouse (Star Schema)"]
        F1["dw.dim_date"]
        F2["dw.dim_location (PostGIS Point EPSG:4326)"]
        F3["dw.dim_road"]
        F4["dw.dim_weather"]
        F5["dw.dim_severity"]
        F6["dw.fact_accidents"]
    end

    subgraph Layer7["Layer 7: Analytical Mart & Dashboard"]
        G1["mart.monthly_hourly_accident_summary"]
        G2["mart.vw_accident_analytics"]
        G3["Streamlit Interactive Dashboard<n(5 Views: Severity, Temporal, Weather, PyDeck Hotspots, Vehicles)"]
    end

    subgraph FutureLayer["Future Scope: Part 2 MLOps Layer (Placeholder)"]
        H1["Feature Store / Feature Engineering"]
        H2["Model Training Pipeline (XGBoost / LightGBM Severity Classifier)"]
        H3["Model Registry & MLflow Tracking"]
        H4["Real-Time Severity Inference API & Monitoring"]
    end

    %% Flow Dependencies
    A1 --> B1
    A2 --> B2
    A3 -.-> B1
    B1 --> B3 --> C1
    B2 --> B3 --> C2
    C1 --> D1
    C2 --> D1
    D1 --> D2
    D2 -->|Invalid Records| D3
    D2 -->|Valid Records| E1 & E2
    E1 & E2 --> D4
    D4 --> F1 & F2 & F3 & F4 & F5
    F1 & F2 & F3 & F4 & F5 --> F6
    F6 --> G1 & G2
    G1 & G2 --> G3
    F6 -.->|Part 2 Integration Point| H1
    H1 -.-> H2 -.-> H3 -.-> H4
```

---

## 7-Layer Architectural Summary

1. **Source Layer**: Official UK DfT STATS19 Road Collision Open Data CSVs and Open-Meteo REST API.
2. **Ingestion Layer**: Modular Python ingestion scripts logging record counts, extraction timestamps, and SHA256 integrity hashes.
3. **Raw Landing Zone**: Local file storage preserving raw unmutated data (`data/raw/`).
4. **Orchestration & ETL Layer**: Apache Airflow DAG (`dags/accident_pipeline_dag.py`) executing a 3-task dependency graph: `extract_accidents` -> `extract_weather` -> `validate_and_load`. Spatial validation rejects out-of-bound coordinates to `data/logs/error_records.log`.
5. **Staging Layer**: Validated CSV datasets with normalized schema types (`data/staging/`).
6. **Storage Layer (Star Schema DW)**: PostgreSQL / PostGIS database hosting dimension tables (`dim_date`, `dim_location`, `dim_road`, `dim_weather`, `dim_severity`) and centralized `fact_accidents` table.
7. **Analytics & Presentation Layer**: Pre-aggregated `mart.monthly_hourly_accident_summary` table and `mart.vw_accident_analytics` view consumed by a 5-tab Streamlit dashboard (`app.py`).
8. **Part 2 MLOps Layer Placeholder**: Decoupled schema interface ready to feed historical features (`fact_accidents` + `dim_weather` + `dim_road`) into Part 2 model training pipelines.
