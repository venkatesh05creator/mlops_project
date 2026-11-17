import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import get_logger

logger = get_logger("ml_features")

# Target encoding mapping
SEVERITY_LABEL_TO_INT = {
    "Slight": 0,
    "Serious": 1,
    "Fatal": 2
}
SEVERITY_INT_TO_LABEL = {v: k for k, v in SEVERITY_LABEL_TO_INT.items()}

# London Reference Coordinates (Charing Cross)
LONDON_CENTER_LAT = 51.5074
LONDON_CENTER_LON = -0.1278


class TrafficFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that engineers domain-specific features
    from raw traffic accident attributes.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def _haversine_np(self, lat1, lon1, lat2, lon2):
        """Vectorized Haversine distance in kilometers."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        km = 6367.0 * c
        return km

    def transform(self, X):
        df = X.copy()
        
        # Ensure timestamp/date types if present
        if "accident_timestamp" in df.columns:
            ts = pd.Series(pd.to_datetime(df["accident_timestamp"]))
            if "accident_hour" not in df.columns:
                df["accident_hour"] = ts.dt.hour
            if "day_of_week" not in df.columns:
                df["day_of_week"] = ts.dt.dayofweek + 1
            if "month" not in df.columns:
                df["month"] = ts.dt.month
            if "is_weekend" not in df.columns:
                df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        # 1. Cyclical Time Features
        hour = df["accident_hour"].fillna(12).astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        dow = df["day_of_week"].fillna(3).astype(float)
        df["day_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * dow / 7.0)

        month = df["month"].fillna(6).astype(float)
        df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

        # 2. Behavioral & Rush-hour Timing
        is_wknd = df["is_weekend"].astype(int) if "is_weekend" in df.columns else (dow >= 6).astype(int)
        df["is_weekend"] = is_wknd
        df["is_rush_hour"] = (((hour.between(7, 9) | hour.between(16, 19))) & (is_wknd == 0)).astype(int)
        df["is_night"] = ((hour < 6) | (hour >= 21)).astype(int)

        # 3. Speed & Kinetic Energy Features
        speed = df["speed_limit"].fillna(30).astype(float)
        df["speed_limit"] = speed
        df["is_high_speed"] = (speed >= 50).astype(int)
        df["kinetic_energy_proxy"] = (speed / 10.0) ** 2

        # 4. Road Surface & Weather Friction Hazards
        surface = df["road_surface"].fillna("Dry").astype(str)
        slippery_keywords = ["wet", "damp", "snow", "ice", "frost", "flood"]
        df["is_slippery_surface"] = surface.apply(
            lambda s: 1 if any(k in s.lower() for k in slippery_keywords) else 0
        )

        temp = df["temperature_celsius"].fillna(10.0).astype(float)
        df["temperature_celsius"] = temp
        df["is_freezing"] = (temp <= 0.0).astype(int)

        precip = df["precipitation_mm"].fillna(0.0).astype(float)
        df["precipitation_mm"] = precip
        df["is_raining"] = (precip > 0.0).astype(int)

        # 5. Visibility & Lighting Hazards
        light = df["light_conditions"].fillna("Daylight").astype(str)
        df["is_dark"] = light.apply(lambda s: 1 if "dark" in s.lower() else 0)
        df["is_dark_unlit"] = light.apply(
            lambda s: 1 if ("no light" in s.lower() or "unlit" in s.lower()) else 0
        )

        # 6. Compound Hazardous Interactions
        df["compound_adverse_condition"] = (
            (df["is_dark"] == 1) & 
            ((df["is_slippery_surface"] == 1) | (df["is_raining"] == 1) | (df["is_freezing"] == 1))
        ).astype(int)
        df["high_speed_in_dark"] = ((df["is_high_speed"] == 1) & (df["is_dark"] == 1)).astype(int)

        # 7. Spatial Geography (Distance from Central London)
        lat = df["latitude"].fillna(LONDON_CENTER_LAT).astype(float)
        lon = df["longitude"].fillna(LONDON_CENTER_LON).astype(float)
        df["latitude"] = lat
        df["longitude"] = lon
        df["dist_from_london_center_km"] = self._haversine_np(
            lat, lon, LONDON_CENTER_LAT, LONDON_CENTER_LON
        )

        # 8. Incident Complexity
        if "number_of_vehicles" in df.columns:
            vehicles = df["number_of_vehicles"].fillna(1).astype(int)
        else:
            vehicles = pd.Series(1, index=df.index, dtype=int)
        df["number_of_vehicles"] = vehicles
        df["is_multi_vehicle"] = (vehicles > 1).astype(int)

        # Retain normalized categoricals for one-hot encoding
        df["road_class"] = df["road_class"].fillna("A").astype(str)
        df["road_surface"] = surface
        df["light_conditions"] = light
        df["weather_condition"] = df["weather_condition"].fillna("Mainly clear").astype(str)

        return df


def get_preprocessor():
    """
    Returns a ColumnTransformer combining feature scaling for numerics
    and one-hot encoding for categoricals.
    """
    numeric_features = [
        "hour_sin", "hour_cos",
        "day_sin", "day_cos",
        "month_sin", "month_cos",
        "is_weekend", "is_rush_hour", "is_night",
        "speed_limit", "is_high_speed", "kinetic_energy_proxy",
        "is_slippery_surface", "temperature_celsius", "is_freezing",
        "precipitation_mm", "is_raining",
        "is_dark", "is_dark_unlit",
        "compound_adverse_condition", "high_speed_in_dark",
        "latitude", "longitude", "dist_from_london_center_km",
        "number_of_vehicles", "is_multi_vehicle"
    ]

    categorical_features = [
        "road_class",
        "road_surface",
        "light_conditions",
        "weather_condition"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


def create_chronological_splits(df: pd.DataFrame):
    """
    Splits analytical dataset chronologically into Train, Validation, and Test sets.
    
    Split Rationale:
    - Train: Jan 1, 2022 to Aug 31, 2022 (Months 1-8, ~67% of data).
      Covers winter, spring, and summer seasons providing diverse environmental baselines.
    - Validation: Sep 1, 2022 to Oct 31, 2022 (Months 9-10, ~17% of data).
      Used for model hyperparameter tuning, imbalance strategy comparison, and early stopping.
    - Test: Nov 1, 2022 to Dec 31, 2022 (Months 11-12, ~16% of data).
      Out-of-time evaluation set featuring late autumn and peak winter hazardous conditions.
    """
    df_sorted = df.sort_values("full_date").copy()
    
    train_mask = df_sorted["full_date"] < "2022-09-01"
    val_mask = (df_sorted["full_date"] >= "2022-09-01") & (df_sorted["full_date"] < "2022-11-01")
    test_mask = df_sorted["full_date"] >= "2022-11-01"

    train_df = df_sorted[train_mask].copy()
    val_df = df_sorted[val_mask].copy()
    test_df = df_sorted[test_mask].copy()

    logger.info(f"Chronological Split Complete:")
    logger.info(f"  Train:      {len(train_df):5d} records ({train_df['full_date'].min().date()} to {train_df['full_date'].max().date()})")
    logger.info(f"  Validation: {len(val_df):5d} records ({val_df['full_date'].min().date()} to {val_df['full_date'].max().date()})")
    logger.info(f"  Test:       {len(test_df):5d} records ({test_df['full_date'].min().date()} to {test_df['full_date'].max().date()})")

    return train_df, val_df, test_df
