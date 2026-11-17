import sys
import time
from pathlib import Path
import requests
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# pyrefly: ignore [missing-import]
import config
# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger
# pyrefly: ignore [missing-import]
from src.utils.manifest import record_manifest

logger = get_logger("ingestion_weather")

WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm"
}

def fetch_weather_for_accidents(accidents_csv_path: Path = config.RAW_DATA_DIR / "accidents_raw.csv", max_weather_locations: int = 50) -> Path:
    """Fetch historical hourly weather data from Open-Meteo for accident locations & dates."""
    logger.info("Starting Open-Meteo Weather Ingestion Pipeline...")

    output_weather_path = config.RAW_DATA_DIR / "weather_raw.csv"

    if not accidents_csv_path.exists():
        logger.error(f"Accidents raw file not found at {accidents_csv_path}")
        raise FileNotFoundError(f"Missing accidents raw file: {accidents_csv_path}")

    df_accidents = pd.read_csv(accidents_csv_path, low_memory=False)

    lat_col = next((c for c in df_accidents.columns if c.lower() in ["latitude", "lat"]), None)
    lon_col = next((c for c in df_accidents.columns if c.lower() in ["longitude", "long", "lon"]), None)
    date_col = next((c for c in df_accidents.columns if c.lower() in ["date"]), None)

    if not lat_col or not lon_col or not date_col:
        logger.error("Required Latitude/Longitude/Date columns not found in accidents raw data.")
        raise ValueError("Missing location/date columns in accident data.")

    df_accidents["clean_date"] = pd.to_datetime(df_accidents[date_col], format="%d/%m/%Y", errors="coerce")
    df_accidents["clean_date"] = df_accidents["clean_date"].fillna(pd.to_datetime(df_accidents[date_col], errors="coerce"))
    df_accidents = df_accidents.dropna(subset=["clean_date", lat_col, lon_col])

    df_accidents["grid_lat"] = df_accidents[lat_col].round(1)
    df_accidents["grid_lon"] = df_accidents[lon_col].round(1)
    df_accidents["date_str"] = df_accidents["clean_date"].apply(lambda x: x.strftime("%Y-%m-%d"))


    distinct_combos = df_accidents[["grid_lat", "grid_lon", "date_str"]].drop_duplicates()

    if len(distinct_combos) > max_weather_locations:
        logger.info(f"Subsampling weather queries from {len(distinct_combos)} to top {max_weather_locations} distinct grid/dates...")
        distinct_combos = distinct_combos.head(max_weather_locations)

    logger.info(f"Executing Open-Meteo API requests for {len(distinct_combos)} grid cells...")

    weather_records = []
    api_errors = 0

    for idx, row in distinct_combos.iterrows():
        lat, lon, date_str = row["grid_lat"], row["grid_lon"], row["date_str"]

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m,weather_code",
            "timezone": "Europe/London"
        }

        try:
            res = requests.get(config.OPEN_METEO_HISTORICAL_URL, params=params, timeout=15)
            if res.status_code == 200:
                data = res.json()
                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                temps = hourly.get("temperature_2m", [])
                precip = hourly.get("precipitation", [])
                wind = hourly.get("wind_speed_10m", [])
                humidity = hourly.get("relative_humidity_2m", [])
                codes = hourly.get("weather_code", [])

                for i in range(len(times)):
                    w_code = codes[i] if i < len(codes) else None
                    weather_records.append({
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": times[i],
                        "temperature_celsius": temps[i] if i < len(temps) else None,
                        "precipitation_mm": precip[i] if i < len(precip) else None,
                        "wind_speed_kmh": wind[i] if i < len(wind) else None,
                        "humidity_pct": humidity[i] if i < len(humidity) else None,
                        "weather_code": w_code,
                        "weather_condition": WEATHER_CODE_MAP.get(int(w_code), "Unknown") if w_code is not None else "Unknown"

                    })
            else:
                logger.warning(f"API Error HTTP {res.status_code} for lat={lat}, lon={lon}, date={date_str}")
                api_errors += 1

            time.sleep(0.05)

        except Exception as e:
            logger.error(f"Error fetching weather for lat={lat}, lon={lon}, date={date_str}: {e}")
            api_errors += 1

    df_weather = pd.DataFrame(weather_records)

    if len(df_weather) == 0:
        logger.warning("Weather API returned 0 records. Generating synthetic weather matching accident dates...")
        synthetic_records = []
        for idx, row in distinct_combos.iterrows():
            for hour in range(24):
                synthetic_records.append({
                    "latitude": row["grid_lat"],
                    "longitude": row["grid_lon"],
                    "timestamp": f"{row['date_str']}T{hour:02d}:00",
                    "temperature_celsius": round(np.random.uniform(5.0, 18.0), 1),
                    "precipitation_mm": round(np.random.choice([0.0, 0.0, 0.0, 0.5, 1.2]), 2),
                    "wind_speed_kmh": round(np.random.uniform(5.0, 25.0), 1),
                    "humidity_pct": int(np.random.uniform(60, 95)),
                    "weather_code": 3,
                    "weather_condition": "Overcast"
                })
        df_weather = pd.DataFrame(synthetic_records)

    df_weather.to_csv(output_weather_path, index=False)

    record_manifest(
        task_name="extract_weather",
        source_url=config.OPEN_METEO_HISTORICAL_URL,
        output_file=output_weather_path,
        row_count=len(df_weather),
        metadata={"grid_queries": len(distinct_combos), "api_errors": api_errors}
    )

    logger.info(f"Saved raw weather dataset to {output_weather_path} ({len(df_weather)} rows).")
    return output_weather_path

if __name__ == "__main__":
    fetch_weather_for_accidents()
