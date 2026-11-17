# PPT Content — Traffic Accident Severity Analytics (Part 1)

Copy each slide's bullets straight into PowerPoint / Google Slides.

---

## Slide 1 — Title
- **Traffic Accident Severity Analytics**
- Data Engineering & MLOps Course Project — Part 1
- End-to-end pipeline: Ingestion → Raw → Cleaning → PostgreSQL/PostGIS Star Schema → Mart → Streamlit Dashboard
- Tech stack: Python (Pandas) · Apache Airflow · PostgreSQL 16 + PostGIS · Streamlit

## Slide 2 — Problem & Objective
- Road accidents are influenced by *when* and *in what conditions* they happen (hour, day, weather, road surface).
- **Goal:** build a fully working, automated data pipeline that joins official UK accident data with historical weather and serves it through an interactive dashboard.
- **Scope (Part 1):** data engineering only — no ML/prediction (that is Part 2).

## Slide 3 — Architecture (7 Layers)
1. **Source** — UK DfT STATS19 collision CSV + Open-Meteo Historical Weather API
2. **Ingestion** — Python scripts with logging & manifests
3. **Raw landing zone** — `data/raw/` (untouched source copies)
4. **Orchestration / ETL** — Airflow DAG (3 tasks) + CLI runner
5. **Staging** — cleaned, validated Parquet files in `data/staging/`
6. **Storage** — PostgreSQL/PostGIS star schema (`analytics`) + aggregate mart (`mart`)
7. **Presentation** — Streamlit dashboard with 5 analytical views
- *(Placeholder for Part 2: MLOps / model training layer)*

## Slide 4 — Data Sources
- **UK DfT STATS19** (Open Government Licence, no PII): national collision file ~106,000 rows (~19 MB); filtered to Police Force 1 (Metropolitan Police / Greater London) → **23,327 accidents**.
- **Open-Meteo Archive API** (free, no API key, hourly ERA5 reanalysis): temperature, precipitation, wind speed, humidity, WMO weather code.
- Coordinates rounded to a 0.1° grid → only ~50 unique location/date queries; **0 API errors**.

## Slide 5 — Ingestion Layer
- **Task 1 — extract_accidents:** stream-download the DfT CSV, filter to Greater London, save `data/raw/accidents_raw.csv`.
- **Task 2 — extract_weather:** for each accident grid cell + date, call Open-Meteo for all 24 hourly records; save `data/raw/weather_raw.csv`.
- **Audit trail:** every run logs timestamp, source URL, row count, file size, SHA-256 hash (JSON manifest per task).

## Slide 6 — Cleaning & Validation
- Reject rows with **missing/out-of-UK-bounds coordinates** (49.8–60.9 lat, −8.6–1.8 lon).
- **Deduplicate** on `accident_index`.
- Parse `DD/MM/YYYY` dates and `HH:MM`/`HHMM` times → ISO timestamps.
- Decode STATS19 integer codes → labels (severity, road type, weather, light, surface) via official DfT lookup tables.
- **Every rejected row is logged with a reason** → `data/logs/rejections.csv`.

## Slide 7 — Data Model: Star Schema + PostGIS
- **Fact:** `analytics.fact_accident` — one row per accident (FKs to all dims, lat/lon + PostGIS `GEOMETRY(Point, 4326)`).
- **Dimensions:** `dim_date`, `dim_location` (PostGIS point + GiST index), `dim_road`, `dim_weather`, `dim_severity`.
- **Join logic:** accidents ↔ weather matched on grid cell (0.1°) + local hour (spatio-temporal join).
- **Mart:** `mart.accident_summary` — pre-aggregated counts by year/month/hour/day/severity/road/weather (**5,104 rows**) for fast dashboards.

## Slide 8 — Orchestration: 3-Task Airflow DAG
- `extract_accidents` → `extract_weather` → `validate_and_load` (clean, join, load DB + mart).
- Same Python functions power both **Airflow** (`dags/`) and a **CLI runner** (`src/pipeline_runner.py`) — the DAG is real and runnable.
- `docker-compose.yml` provisions PostgreSQL/PostGIS; a portable PG16+PostGIS setup script covers machines without Docker.

## Slide 9 — Streamlit Dashboard (5 Views)
1. **Severity distribution** — Fatal/Serious/Slight breakdown (~86% slight, ~13% serious, ~1% fatal)
2. **Accidents by hour & weekday** — rush-hour peaks 08:00–09:00 and 17:00–18:30, weekday×hour matrix
3. **Weather & road impact** — dry vs wet roads, clear vs rain conditions
4. **Location hotspot map** — PyDeck geospatial scatter over London
5. **Vehicle & casualty analysis** — distributions per accident
- KPI header (accidents, casualties, vehicles, serious/fatal %) + sidebar filters.

## Slide 10 — Engineering Practices
- All credentials via `.env` (never hardcoded); no PII; OGL-licensed open data.
- Run manifests (SHA-256, row counts) + rejection audit log = full lineage & auditability.
- Structured logging to console + `data/logs/*.log`.
- Unit tests (`pytest`) cover UK bounding-box rejection, code decoding, config paths.

## Slide 11 — Results & Part 2 Roadmap
- **Delivered:** 23,327 fact rows ingested→validated→loaded; 720 hourly weather records; full star schema + mart; 5-view dashboard; 3-task DAG.
- **Part 2 (MLOps):** feature store from fact+weather+road → train severity classifier (XGBoost/LightGBM) → MLflow tracking → FastAPI inference API → monitoring.

---

## File-by-File Explanation (for backup/appendix slides)

### Root
| File | What it does |
|---|---|
| `config.py` | Single source of truth: loads `.env`, DB connection URL builder, data folder paths, DfT download URLs, police-force filter, UK bounding box, Open-Meteo endpoint. |
| `.env` / `.env.example` | Environment variables (DB host/port/user/password/name, police force filter, SQLite fallback flag) — no secrets in code. |
| `app.py` | **Streamlit dashboard.** Loads mart/view data from the DB (cached 5 min), 4 KPI metrics, sidebar filters (severity, road surface), and 5 tabs: severity, hour×weekday, weather/road, PyDeck hotspot map, vehicles/casualties. |
| `requirements.txt` | Pinned dependencies: pandas, requests, SQLAlchemy, psycopg, streamlit, pydeck, pytest, python-dotenv, pyarrow. |
| `docker-compose.yml` | Provisions PostgreSQL + PostGIS container (port from `.env`, health check, auto-applies `sql/01_init_schema.sql`). |
| `README.md` | Setup guide: portable Postgres script, tests, pipeline run, dashboard launch. |

### `src/` — pipeline code
| File | What it does |
|---|---|
| `src/pipeline_runner.py` | CLI orchestration of the same 3 tasks as the DAG: extract accidents → extract weather → clean+join+load; logs total runtime. |
| `src/ingestion/download_accidents.py` | **Task 1.** Streams the DfT CSV, filters to Police Force 1 (Greater London), writes `data/raw/accidents_raw.csv`, computes SHA-256, logs counts/errors. |
| `src/ingestion/fetch_weather.py` | **Task 2.** Rounds accident coords to a 0.1° grid, queries Open-Meteo hourly archive per cell+date (max 50 queries), decodes WMO weather codes, writes `data/raw/weather_raw.csv`, records manifest with API-error count. |
| `src/transformation/clean_and_validate.py` | **Task 3a.** Row-by-row validation: UK bounding box, dedup, date/time parsing, code→label decoding; exports clean Parquet to staging and logs every rejected row with its reason. |
| `src/transformation/join_and_load.py` | **Task 3b.** Spatio-temporal join (grid cell + floor-hour) of accidents with weather, applies DDL, loads staging → dims (`dim_severity`, `dim_road`, `dim_weather`) → `fact_accident`, fills PostGIS `ST_MakePoint` geometries, aggregates and loads `mart.accident_summary`. |
| `src/transformation/lookups.py` | Official DfT STATS19 code→label mappings (severity, weather, road type, light, surface, urban/rural) + safe `decode_attribute()` helper. |
| `src/utils/logger.py` | `get_logger()` — structured logging to console and `data/logs/pipeline.log`. |
| `src/utils/manifest.py` | Run manifests (task, UTC timestamp, source URL, output path, size, SHA-256, row count) as JSON per task; appends rejected rows to `data/logs/rejections.csv`. |

### `dags/`, `sql/`, `scripts/`, `tests/`
| File | What it does |
|---|---|
| `dags/accident_pipeline_dag.py` | Airflow DAG `traffic_accident_severity_pipeline` with 3 PythonOperator tasks chained `extract_accidents >> extract_weather >> validate_and_load` (monthly schedule, retries). |
| `sql/01_init_schema.sql` | DDL: PostGIS extension; schemas `raw/staging/dw/mart`; star-schema tables + indexes; `mart.vw_accident_analytics` view joining all dims for the dashboard. |
| `scripts/setup_postgres_runtime.py` | No-Docker setup: extracts portable PostgreSQL 16 binaries + PostGIS bundle into `runtime/`, initializes the cluster, starts it on port 5433, creates the `traffic` DB and PostGIS extension. |
| `tests/test_pipeline.py` | pytest suite: out-of-UK coordinate is rejected, severity codes decode correctly (1/2/3/unknown), config directories exist. |

### `data/` — pipeline outputs (evidence for screenshots)
| Path | Contents |
|---|---|
| `data/raw/` | `accidents_raw.csv` (filtered source data), `weather_raw.csv` (hourly weather rows). |
| `data/staging/` | Cleaned Parquet files (`accidents_clean.parquet`, `weather_clean.parquet`). |
| `data/logs/` | Task logs, `rejections.csv` (rejected rows + reasons), JSON run manifests, `traffic_analytics.db` (SQLite fallback mode). |

### `docs/`
| File | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Architecture rationale, 1-way data flow diagram, state-ownership table. |
| `docs/architecture_diagram.md` | Mermaid 7-layer diagram incl. Part 2 MLOps placeholder. |
| `docs/DATA_DICTIONARY.md` (+ `data_dictionary.md` copy) | Every staging/analytics/mart column with type and description. |
| `docs/REPORT.md` | Full project report: architecture, validation rules, PostGIS, findings. |
| `docs/final_report.md` | Long-form technical report with run numbers (23,327 accidents; 720 weather rows; 5,104 mart rows). |
| `docs/screenshot_evidence_guide.md` | The 6 screenshots to capture as proof of an end-to-end run. |
| `docs/PPT_CONTENT.md` | This file. |
