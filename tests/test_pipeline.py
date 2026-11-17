import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
import config
# pyrefly: ignore [missing-import]
from src.transformation.clean_and_validate import clean_accidents
# pyrefly: ignore [missing-import]
from src.transformation.lookups import SEVERITY_MAP, decode_attribute

def test_uk_bounding_box_validation():
    """Verify that coordinates outside UK bounds are properly rejected."""
    df_dummy = pd.DataFrame([
        {
            "accident_index": "ACC_VALID",
            "latitude": 51.5074,
            "longitude": -0.1278,
            "date": "14/05/2025",
            "time": "14:30",
            "accident_severity": 3
        },
        {
            "accident_index": "ACC_INVALID_LAT",
            "latitude": 35.6895,  # Tokyo Lat (Out of UK bounds)
            "longitude": -0.1278,
            "date": "14/05/2025",
            "time": "14:30",
            "accident_severity": 3
        }
    ])

    df_clean, df_rejected = clean_accidents(df_dummy)

    assert len(df_clean) == 1
    assert df_clean.iloc[0]["accident_index"] == "ACC_VALID"
    assert len(df_rejected) == 1
    assert "out of UK bounds" in df_rejected.iloc[0]["reason"]

def test_severity_lookup_decodes():
    """Verify DfT STATS19 severity code decodes."""
    assert decode_attribute(1, SEVERITY_MAP) == "Fatal"
    assert decode_attribute(2, SEVERITY_MAP) == "Serious"
    assert decode_attribute(3, SEVERITY_MAP) == "Slight"
    assert decode_attribute(99, SEVERITY_MAP, default="Unknown") == "Unknown"

def test_config_paths_exist():
    """Verify critical data directories are properly initialized."""
    assert config.RAW_DATA_DIR.exists()
    assert config.STAGING_DATA_DIR.exists()
    assert config.LOGS_DIR.exists()
