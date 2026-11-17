import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils.logger import get_logger

logger = get_logger("ml_data_loader")


def load_analytical_data(db_url: str = None) -> pd.DataFrame:
    """
    Extracts accident analytical mart records from PostgreSQL DW/Mart or SQLite fallback.
    Records are chronologically sorted by accident_timestamp.
    """
    if db_url is None:
        db_url = config.get_db_url()

    logger.info(f"Connecting to analytical database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    engine = create_engine(db_url)
    is_sqlite = config.USE_SQLITE_FALLBACK or "sqlite" in db_url

    if is_sqlite:
        query = """
        SELECT 
            f.accident_index,
            f.accident_timestamp,
            f.accident_hour,
            d.full_date,
            d.year,
            d.month,
            d.month_name,
            d.day_name,
            d.day_of_week,
            d.is_weekend,
            l.latitude,
            l.longitude,
            l.police_force_name,
            r.road_class,
            r.road_surface,
            r.speed_limit,
            r.light_conditions,
            w.weather_condition,
            w.temperature_celsius,
            w.precipitation_mm,
            s.accident_severity_label,
            s.number_of_vehicles,
            s.number_of_casualties
        FROM fact_accidents f
        JOIN dim_date d ON f.date_key = d.date_key
        JOIN dim_location l ON f.location_key = l.location_key
        JOIN dim_road r ON f.road_key = r.road_key
        LEFT JOIN dim_weather w ON f.weather_key = w.weather_key
        JOIN dim_severity s ON f.severity_key = s.severity_key
        ORDER BY f.accident_timestamp ASC
        """
    else:
        query = """
        SELECT 
            accident_index,
            accident_timestamp,
            accident_hour,
            full_date,
            year,
            month,
            month_name,
            day_name,
            day_of_week,
            is_weekend,
            latitude,
            longitude,
            police_force_name,
            road_class,
            road_surface,
            speed_limit,
            light_conditions,
            weather_condition,
            temperature_celsius,
            precipitation_mm,
            accident_severity_label,
            number_of_vehicles,
            number_of_casualties
        FROM mart.vw_accident_analytics
        ORDER BY accident_timestamp ASC
        """

    df = pd.read_sql(query, engine)
    df["accident_timestamp"] = pd.to_datetime(df["accident_timestamp"])
    df["full_date"] = pd.to_datetime(df["full_date"])
    
    logger.info(f"Loaded {len(df)} records spanning {df['full_date'].min().date()} to {df['full_date'].max().date()}")
    return df


if __name__ == "__main__":
    df = load_analytical_data()
    print("Loaded data shape:", df.shape)
    print("Severity distribution:\n", df["accident_severity_label"].value_counts())
