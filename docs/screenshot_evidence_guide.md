# Screenshot & Execution Evidence Capture Guide

To prove a 100% successful end-to-end pipeline run for course submission, capture the following 6 core evidence artifacts:

---

## 📸 1. Raw Data Ingestion & Audit Logs
- **What to capture**: Terminal output of `python src/pipeline_runner.py` showing **Step 1 & Step 2** completion.
- **Key details to highlight**:
  - Downloaded raw CSV file size (~19.18 MB).
  - Filtered record count (`23,327` records for Metropolitan Police).
  - SHA256 checksum hash digest.
  - Weather API response count (`720` hourly rows, `0` errors).
- **Log file reference**: `data/logs/ingestion_accidents.log` and `data/logs/ingestion_weather.log`.

---

## 📸 2. Data Validation & Rejection Log
- **What to capture**: Terminal output and text file content of `data/logs/error_records.log` and `data/logs/transformation_cleaning.log`.
- **Key details to highlight**:
  - Total accepted records (`23,327`).
  - Validation log showing UK bounding box check ($49.8 \le \text{lat} \le 60.9$, $-8.6 \le \text{lon} \le 1.8$).

---

## 📸 3. Database DW & Mart Schema Verification
- **What to capture**: SQL client (e.g. DBeaver, `psql`, or SQLite Browser) or Python count output showing populated Star Schema tables.
- **Key details to highlight**:
  - `dw.fact_accidents`: **23,327** rows.
  - `dw.dim_date`, `dw.dim_location`, `dw.dim_road`, `dw.dim_weather`, `dw.dim_severity`.
  - `mart.monthly_hourly_accident_summary`: **5,104** rows.

---

## 📸 4. Airflow DAG Workflow Execution
- **What to capture**: Apache Airflow UI Grid/Tree view showing the 3 green success status boxes for DAG `traffic_accident_severity_pipeline`.
- **Key details to highlight**:
  - Sequential task flow: `extract_accidents` $\rightarrow$ `extract_weather` $\rightarrow$ `validate_and_load`.

---

## 📸 5. Streamlit Analytical Dashboard (5 Views)
- **What to capture**: Browser screenshots of `http://localhost:8501` showing each of the 5 tabs:
  1. **Tab 1 — Severity Distribution**: Bar chart & KPI metric metrics.
  2. **Tab 2 — Temporal Patterns**: Hourly trend line chart and day-of-week matrix.
  3. **Tab 3 — Weather & Road Impact**: Weather condition & road surface breakdown.
  4. **Tab 4 — Geospatial Hotspot Map**: Interactive PyDeck 3D map showing collision clusters in London.
  5. **Tab 5 — Vehicle & Casualty Analysis**: Bar charts of casualty distribution and vehicle counts.

---

## 📸 6. Unit Test Pass Proof
- **What to capture**: Terminal screenshot running `pytest tests/` showing green passing test status.
