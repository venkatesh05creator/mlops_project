import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# pyrefly: ignore [missing-import]
import config
# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger
# pyrefly: ignore [missing-import]
from src.utils.manifest import log_rejections, record_manifest
# pyrefly: ignore [missing-import]
from src.transformation.lookups import (
    SEVERITY_MAP, WEATHER_MAP, ROAD_TYPE_MAP,
    LIGHT_CONDITIONS_MAP, ROAD_SURFACE_MAP, URBAN_RURAL_MAP, decode_attribute
)

logger = get_logger("clean_and_validate")

def clean_accidents(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and validate accident records against UK spatial bounding box, temporal formats, and completeness constraints."""
    rejected_rows = []
    valid_rows = []

    raw_cols = [str(c) for c in df_raw.columns]
    idx_col = next((c for c in raw_cols if "index" in c.lower() or "reference" in c.lower()), raw_cols[0])
    lat_col = next((c for c in raw_cols if c.lower() in ["latitude", "lat"]), None)
    lon_col = next((c for c in raw_cols if c.lower() in ["longitude", "long", "lon"]), None)
    date_col = next((c for c in raw_cols if c.lower() in ["date"]), None)
    time_col = next((c for c in raw_cols if c.lower() in ["time"]), None)
    sev_col = next((c for c in raw_cols if "severity" in c.lower()), None)

    seen_indices = set()

    for i, row in df_raw.iterrows():
        acc_idx = str(row.get(idx_col, f"ACC_{i}"))

        if acc_idx in seen_indices:
            rejected_rows.append({**dict(row), "accident_index": acc_idx, "reason": "Duplicate accident index"})
            continue
        seen_indices.add(acc_idx)

        # Validate Coordinates
        raw_lat = row.get(lat_col) if lat_col is not None else None
        raw_lon = row.get(lon_col) if lon_col is not None else None

        if raw_lat is None or raw_lon is None or pd.isna(raw_lat) or pd.isna(raw_lon):
            rejected_rows.append({**dict(row), "accident_index": acc_idx, "reason": "Missing or non-numeric coordinates"})
            continue

        try:
            lat = float(raw_lat)
            lon = float(raw_lon)
            if not (config.UK_BOUNDS["min_lat"] <= lat <= config.UK_BOUNDS["max_lat"] and 
                    config.UK_BOUNDS["min_lon"] <= lon <= config.UK_BOUNDS["max_lon"]):
                rejected_rows.append({**dict(row), "accident_index": acc_idx, "reason": f"Coordinates ({lat}, {lon}) out of UK bounds"})
                continue
        except (ValueError, TypeError):
            rejected_rows.append({**dict(row), "accident_index": acc_idx, "reason": "Missing or non-numeric coordinates"})
            continue

        # Validate Date & Time
        raw_date = row.get(date_col)
        raw_time = str(row.get(time_col, "12:00")).strip()

        if pd.isna(raw_date):
            rejected_rows.append({**dict(row), "accident_index": acc_idx, "reason": "Missing accident date"})
            continue

        try:
            parsed_date = pd.to_datetime(raw_date, format="%d/%m/%Y", errors="coerce")
            if pd.isna(parsed_date):
                parsed_date = pd.to_datetime(raw_date, errors="coerce")

            if pd.isna(parsed_date):
                rejected_rows.append({**dict(row), "accident_index": acc_idx, "reason": f"Invalid date format: {raw_date}"})
                continue

            if len(raw_time) == 4 and raw_time.isdigit():
                hour = int(raw_time[:2])
                minute = int(raw_time[2:])
            elif ":" in raw_time:
                parts = raw_time.split(":")
                hour = int(parts[0])
                minute = int(parts[1])
            else:
                hour = 12
                minute = 0

            timestamp = datetime(int(parsed_date.year), int(parsed_date.month), int(parsed_date.day), hour, minute)

        except Exception as ex:
            rejected_rows.append({**dict(row), "accident_index": acc_idx, "reason": f"Failed to parse date/time: {ex}"})
            continue

        raw_sev = row.get(sev_col, 3)
        sev_label = decode_attribute(raw_sev, SEVERITY_MAP, "Slight")

        road_surf_code = row.get("road_surface_conditions", -1)
        road_surf_label = decode_attribute(road_surf_code, ROAD_SURFACE_MAP, "Dry")

        light_code = row.get("light_conditions", -1)
        light_label = decode_attribute(light_code, LIGHT_CONDITIONS_MAP, "Daylight")

        weather_code = row.get("weather_conditions", -1)
        weather_label = decode_attribute(weather_code, WEATHER_MAP, "Fine no high winds")

        road_type_code = row.get("road_type", -1)
        road_type_label = decode_attribute(road_type_code, ROAD_TYPE_MAP, "Single carriageway")

        valid_rows.append({
            "accident_index": acc_idx,
            "accident_timestamp": timestamp,
            "date_key": int(timestamp.strftime("%Y%m%d")),
            "accident_year": timestamp.year,
            "accident_month": timestamp.month,
            "accident_day": timestamp.day,
            "accident_hour": timestamp.hour,
            "day_of_week": timestamp.isoweekday(),
            "day_name": timestamp.strftime("%A"),
            "is_weekend": timestamp.isoweekday() in [6, 7],
            "latitude": lat,
            "longitude": lon,
            "police_force": row.get("police_force", 1),
            "local_authority_district": row.get("local_authority_district", 1),
            "accident_severity_code": raw_sev,
            "accident_severity_label": sev_label,
            "road_type_label": road_type_label,
            "road_surface": road_surf_label,
            "speed_limit": int(row.get("speed_limit", 30)) if pd.notna(row.get("speed_limit")) else 30,
            "light_conditions": light_label,
            "weather_conditions": weather_label,
            "number_of_vehicles": int(row.get("number_of_vehicles", 1)) if pd.notna(row.get("number_of_vehicles")) else 1,
            "number_of_casualties": int(row.get("number_of_casualties", 1)) if pd.notna(row.get("number_of_casualties")) else 1
        })

    df_clean = pd.DataFrame(valid_rows)
    df_rejected = pd.DataFrame(rejected_rows)
    return df_clean, df_rejected

def run_cleaning_pipeline() -> tuple[Path, Path]:
    """Execute cleaning, spatial/temporal validation, rejection auditing, and staging export."""
    logger.info("Starting Data Cleaning & Validation Pipeline...")

    raw_accidents_path = config.RAW_DATA_DIR / "accidents_raw.csv"
    raw_weather_path = config.RAW_DATA_DIR / "weather_raw.csv"

    staging_accidents_path = config.STAGING_DATA_DIR / "accidents_clean.parquet"
    staging_weather_path = config.STAGING_DATA_DIR / "weather_clean.parquet"

    if not raw_accidents_path.exists():
        raise FileNotFoundError(f"Raw accidents CSV missing at {raw_accidents_path}")

    df_raw_accidents = pd.read_csv(raw_accidents_path, low_memory=False)
    logger.info(f"Loaded {len(df_raw_accidents)} raw accident records.")

    df_clean_accidents, df_rejected = clean_accidents(df_raw_accidents)

    if not df_rejected.empty:
        log_rejections(df_rejected, reason="Data Validation Failure")
        logger.warning(f"Audited and logged {len(df_rejected)} rejected records to runtime/logs/rejections.csv")

    logger.info(f"Accepted Records: {len(df_clean_accidents)} | Rejected Records: {len(df_rejected)}")
    df_clean_accidents.to_parquet(staging_accidents_path, index=False)

    if raw_weather_path.exists():
        df_raw_weather = pd.read_csv(raw_weather_path)
        df_raw_weather["timestamp"] = pd.to_datetime(df_raw_weather["timestamp"], errors="coerce")
        df_raw_weather = df_raw_weather.dropna(subset=["timestamp", "latitude", "longitude"])
        df_raw_weather.to_parquet(staging_weather_path, index=False)
        logger.info(f"Saved cleaned weather records to staging: {len(df_raw_weather)}")

    record_manifest(
        task_name="clean_and_validate",
        source_url=str(raw_accidents_path),
        output_file=staging_accidents_path,
        row_count=len(df_clean_accidents),
        metadata={"rejected_count": len(df_rejected)}
    )

    return staging_accidents_path, staging_weather_path

if __name__ == "__main__":
    run_cleaning_pipeline()
