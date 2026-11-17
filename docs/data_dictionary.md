# Data Dictionary

## 1. Staging Schema (`staging`)

### `staging.stg_collision`
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `accident_index` | VARCHAR(50) | Primary Key. Unique identifier for each collision record |
| `accident_timestamp` | TIMESTAMP | Parsed accident date and time |
| `date_key` | INT | Integer representation of date (YYYYMMDD) |
| `accident_year` | INT | Year of collision |
| `accident_month` | INT | Month of collision (1-12) |
| `accident_day` | INT | Day of month (1-31) |
| `accident_hour` | INT | Hour of collision (0-23) |
| `day_of_week` | INT | ISO weekday (1=Monday .. 7=Sunday) |
| `day_name` | VARCHAR(20) | Full day name |
| `is_weekend` | BOOLEAN | True if Saturday or Sunday |
| `latitude` | DOUBLE PRECISION | WGS84 Latitude coordinate |
| `longitude` | DOUBLE PRECISION | WGS84 Longitude coordinate |
| `police_force` | INT | DfT Police Force code |
| `local_authority_district` | INT | Local Authority District code |
| `accident_severity_code` | INT | Raw DfT severity integer (1=Fatal, 2=Serious, 3=Slight) |
| `accident_severity_label` | VARCHAR(20) | Decoded severity text |
| `road_type_label` | VARCHAR(50) | Decoded road classification |
| `road_surface` | VARCHAR(50) | Decoded road surface condition |
| `speed_limit` | INT | Speed limit in mph |
| `light_conditions` | VARCHAR(50) | Decoded lighting conditions |
| `weather_conditions` | VARCHAR(50) | Decoded weather conditions |
| `number_of_vehicles` | INT | Count of vehicles involved |
| `number_of_casualties` | INT | Count of casualties reported |

---

## 2. Analytics Star Schema (`analytics`)

### `analytics.fact_accident`
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `accident_id` | SERIAL (PK) | Surrogate primary key |
| `accident_index` | VARCHAR(50) | Unique collision index |
| `accident_timestamp` | TIMESTAMP | Exact collision timestamp |
| `date_key` | INT | Foreign key to date |
| `severity_key` | INT | Foreign key to `analytics.dim_severity` |
| `road_key` | INT | Foreign key to `analytics.dim_road` |
| `weather_key` | INT | Foreign key to `analytics.dim_weather` |
| `latitude` | DOUBLE PRECISION | WGS84 Latitude |
| `longitude` | DOUBLE PRECISION | WGS84 Longitude |
| `geom` | GEOMETRY(Point, 4326) | PostGIS spatial point geometry (SRID 4326) |
| `number_of_vehicles` | INT | Total vehicles involved |
| `number_of_casualties` | INT | Total casualties |

---

## 3. Data Mart Schema (`mart`)

### `mart.accident_summary`
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `summary_id` | SERIAL (PK) | Data mart summary ID |
| `accident_year` | INT | Aggregated year |
| `accident_month` | INT | Aggregated month |
| `accident_hour` | INT | Aggregated hour of day |
| `day_name` | VARCHAR(20) | Day of week name |
| `is_weekend` | BOOLEAN | Weekend indicator |
| `severity_label` | VARCHAR(20) | Severity classification |
| `road_type_label` | VARCHAR(50) | Road classification |
| `weather_condition` | VARCHAR(50) | ERA5 weather condition |
| `total_accidents` | INT | Total count of collisions |
| `total_vehicles` | INT | Sum of vehicles involved |
| `total_casualties` | INT | Sum of casualties |
