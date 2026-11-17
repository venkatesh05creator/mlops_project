import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from src.utils.logger import get_logger
from src.utils.manifest import record_manifest

logger = get_logger("join_and_load")

def run_dw_load_pipeline(
    staging_accidents_path: Path = config.STAGING_DATA_DIR / "accidents_clean.parquet",
    staging_weather_path: Path = config.STAGING_DATA_DIR / "weather_clean.parquet"
):
    """Executes spatio-temporal join of accidents + weather, schema DDL initialization, PostGIS loading, and mart creation."""
    logger.info("Starting Spatio-Temporal Join & PostgreSQL/PostGIS Load Pipeline...")

    if not staging_accidents_path.exists():
        raise FileNotFoundError(f"Missing cleaned accidents staging file: {staging_accidents_path}")

    df_accidents = pd.read_parquet(staging_accidents_path)
    df_accidents["accident_timestamp"] = pd.to_datetime(df_accidents["accident_timestamp"])

    df_weather = pd.DataFrame()
    if staging_weather_path.exists():
        df_weather = pd.read_parquet(staging_weather_path)
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])

    df_accidents["grid_lat"] = df_accidents["latitude"].round(1)
    df_accidents["grid_lon"] = df_accidents["longitude"].round(1)
    df_accidents["match_time"] = df_accidents["accident_timestamp"].dt.floor("h")

    if not df_weather.empty:
        df_weather["grid_lat"] = df_weather["latitude"].round(1)
        df_weather["grid_lon"] = df_weather["longitude"].round(1)
        df_weather["match_time"] = df_weather["timestamp"].dt.floor("h")

        logger.info("Performing spatio-temporal join (grid cell + local hour)...")
        df_merged = pd.merge(
            df_accidents,
            df_weather,
            on=["grid_lat", "grid_lon", "match_time"],
            how="left",
            suffixes=("", "_weather")
        )
    else:
        df_merged = df_accidents.copy()
        df_merged["temperature_celsius"] = 12.0
        df_merged["precipitation_mm"] = 0.0
        df_merged["wind_speed_kmh"] = 10.0
        df_merged["humidity_pct"] = 75
        df_merged["weather_code"] = 1
        df_merged["weather_condition"] = "Mainly clear"

    df_merged["temperature_celsius"] = df_merged["temperature_celsius"].fillna(12.0)
    df_merged["precipitation_mm"] = df_merged["precipitation_mm"].fillna(0.0)
    df_merged["wind_speed_kmh"] = df_merged["wind_speed_kmh"].fillna(10.0)
    df_merged["humidity_pct"] = df_merged["humidity_pct"].fillna(75)
    df_merged["weather_condition"] = df_merged["weather_condition"].fillna("Mainly clear")

    logger.info(f"Spatio-temporal join complete. Total records: {len(df_merged)}")

    db_url = config.get_db_url()
    logger.info(f"Connecting to database: {db_url}")
    engine = create_engine(db_url)

    # Initialize SQL Schemas & PostGIS Extension
    sql_script_path = config.BASE_DIR / "sql" / "01_schema.sql"
    with open(sql_script_path, "r", encoding="utf-8") as f:
        sql_ddl = f.read()

    with engine.begin() as conn:
        logger.info("Dropping existing tables with CASCADE for clean pipeline re-run...")
        conn.execute(text("""
            DROP TABLE IF EXISTS mart.accident_summary CASCADE;
            DROP TABLE IF EXISTS analytics.fact_accident CASCADE;
            DROP TABLE IF EXISTS analytics.dim_weather CASCADE;
            DROP TABLE IF EXISTS analytics.dim_road CASCADE;
            DROP TABLE IF EXISTS analytics.dim_severity CASCADE;
            DROP TABLE IF EXISTS staging.stg_collision CASCADE;
            DROP TABLE IF EXISTS staging.stg_weather CASCADE;
        """))

        logger.info("Executing DDL initialization for PostgreSQL/PostGIS schemas...")
        statements = [s.strip() for s in sql_ddl.split(";") if s.strip()]
        for stmt in statements:
            try:
                conn.execute(text(stmt))
            except Exception as ex:
                logger.warning(f"DDL statement notice/error: {ex}")

    # Load Staging Tables (Filtering temporary join columns like grid_lat, grid_lon, match_time)
    stg_acc_cols = [c for c in df_accidents.columns if c not in ["grid_lat", "grid_lon", "match_time"]]
    df_accidents[stg_acc_cols].to_sql("stg_collision", engine, schema="staging", if_exists="append", index=False)
    if not df_weather.empty:
        stg_wea_cols = [c for c in df_weather.columns if c not in ["grid_lat", "grid_lon", "match_time"]]
        df_weather[stg_wea_cols].to_sql("stg_weather", engine, schema="staging", if_exists="append", index=False)

    # Build Dimensions
    # 1. Severity Dimension
    severity_records = [
        {"severity_key": 1, "severity_label": "Fatal"},
        {"severity_key": 2, "severity_label": "Serious"},
        {"severity_key": 3, "severity_label": "Slight"}
    ]
    df_dim_sev = pd.DataFrame(severity_records)
    df_dim_sev.to_sql("dim_severity", engine, schema="analytics", if_exists="append", index=False)

    # 2. Road Dimension
    df_dim_road = df_merged[["road_type_label", "road_surface", "speed_limit", "light_conditions"]].drop_duplicates().reset_index(drop=True)
    df_dim_road["road_key"] = df_dim_road.index + 1
    df_dim_road.to_sql("dim_road", engine, schema="analytics", if_exists="append", index=False)

    # 3. Weather Dimension
    df_dim_weather = df_merged[["grid_lat", "grid_lon", "match_time", "temperature_celsius", "precipitation_mm", "wind_speed_kmh", "humidity_pct", "weather_condition"]].drop_duplicates().reset_index(drop=True)
    df_dim_weather["weather_key"] = df_dim_weather.index + 1
    df_dim_weather = df_dim_weather.rename(columns={"grid_lat": "grid_latitude", "grid_lon": "grid_longitude", "match_time": "weather_timestamp"})
    df_dim_weather.to_sql("dim_weather", engine, schema="analytics", if_exists="append", index=False)

    # Map FK keys back to facts
    df_fact = pd.merge(df_merged, df_dim_road, on=["road_type_label", "road_surface", "speed_limit", "light_conditions"])
    df_fact = pd.merge(
        df_fact,
        df_dim_weather.rename(columns={"grid_latitude": "grid_lat", "grid_longitude": "grid_lon", "weather_timestamp": "match_time"}),
        on=["grid_lat", "grid_lon", "match_time", "temperature_celsius", "precipitation_mm", "wind_speed_kmh", "humidity_pct", "weather_condition"]
    )

    sev_key_map = {"Fatal": 1, "Serious": 2, "Slight": 3}
    df_fact["severity_key"] = df_fact["accident_severity_label"].map(sev_key_map).fillna(3).astype(int)

    df_fact_final = pd.DataFrame({
        "accident_index": df_fact["accident_index"],
        "accident_timestamp": df_fact["accident_timestamp"],
        "date_key": df_fact["date_key"],
        "severity_key": df_fact["severity_key"],
        "road_key": df_fact["road_key"],
        "weather_key": df_fact["weather_key"],
        "latitude": df_fact["latitude"],
        "longitude": df_fact["longitude"],
        "number_of_vehicles": df_fact["number_of_vehicles"],
        "number_of_casualties": df_fact["number_of_casualties"],
        "grid_lat": df_fact["grid_lat"],
        "grid_lon": df_fact["grid_lon"]
    }).drop_duplicates(subset=["accident_index"])

    df_fact_final.to_sql("fact_accident", engine, schema="analytics", if_exists="append", index=False)

    # Populate PostGIS Geometry point column
    with engine.begin() as conn:
        logger.info("Populating PostGIS ST_Point spatial geometries in analytics.fact_accident...")
        conn.execute(text("""
            UPDATE analytics.fact_accident 
            SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            WHERE geom IS NULL;
        """))

    # Build Mart Summary
    df_mart = df_merged.groupby([
        "accident_year", "accident_month", "accident_hour", "day_name", "is_weekend",
        "accident_severity_label", "road_type_label", "weather_condition"
    ]).agg(
        total_accidents=("accident_index", "count"),
        total_vehicles=("number_of_vehicles", "sum"),
        total_casualties=("number_of_casualties", "sum")
    ).reset_index().rename(columns={
        "accident_year": "accident_year",
        "accident_month": "accident_month",
        "accident_severity_label": "severity_label"
    })

    df_mart.to_sql("accident_summary", engine, schema="mart", if_exists="append", index=False)

    record_manifest(
        task_name="validate_and_load",
        source_url="staging",
        output_file=Path(db_url),
        row_count=len(df_fact_final),
        metadata={"fact_count": len(df_fact_final), "mart_count": len(df_mart)}
    )

    logger.info(f"SUCCESS: Loaded {len(df_fact_final)} facts and {len(df_mart)} mart records into PostgreSQL + PostGIS!")

if __name__ == "__main__":
    run_dw_load_pipeline()
