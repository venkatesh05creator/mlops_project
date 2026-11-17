import os
import sys
import logging
import requests
import pandas as pd
import hashlib
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# pyrefly: ignore [missing-import]
import config

# Setup Logging
LOG_FILE = config.LOGS_DIR / "ingestion_accidents.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file for data lineage and integrity check."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def fetch_and_filter_accidents(url: str = config.ACCIDENT_DATA_URL_2022, police_force_filter: int = config.POLICE_FORCE_FILTER) -> Path:
    """Download collision CSV, apply regional filter, and save to raw landing zone."""
    extraction_date = datetime.now().isoformat()
    logging.info(f"--- Starting Accident Data Ingestion Pipeline ---")
    logging.info(f"Extraction Timestamp: {extraction_date}")
    logging.info(f"Source URL: {url}")
    logging.info(f"Police Force Filter ID: {police_force_filter}")

    raw_download_path = config.RAW_DATA_DIR / "accidents_full_temp.csv"
    output_raw_path = config.RAW_DATA_DIR / "accidents_raw.csv"

    try:
        logging.info("Initiating dataset download...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(raw_download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Download complete. Raw file size: {raw_download_path.stat().st_size / (1024*1024):.2f} MB")

        # Load CSV into pandas for filtering & validation
        df = pd.read_csv(raw_download_path, low_memory=False)
        total_uk_records = len(df)
        logging.info(f"Loaded total UK collision records: {total_uk_records}")

        # Filter by Police Force if column exists
        police_col = None
        for col in df.columns:
            if col.lower() in ["police_force", "policeforce"]:
                police_col = col
                break

        if police_col and police_force_filter:
            # Convert to numeric safely
            df[police_col] = pd.to_numeric(df[police_col], errors="coerce")
            df_filtered = df[df[police_col] == police_force_filter].copy()
            logging.info(f"Filtered records for Police Force {police_force_filter}: {len(df_filtered)} records")
        else:
            df_filtered = df.head(10000).copy() # Fallback subset if no police force column
            logging.info(f"No police force column match found. Scoped down to top {len(df_filtered)} records.")

        # Save scoped raw CSV
        df_filtered.to_csv(output_raw_path, index=False)
        
        # Cleanup temp file
        if raw_download_path.exists():
            raw_download_path.unlink()

        file_hash = compute_file_hash(output_raw_path)
        final_count = len(df_filtered)

        logging.info(f"SUCCESS: Raw accidents saved to {output_raw_path}")
        logging.info(f"Audit Summary -> Record Count: {final_count}, SHA256: {file_hash}")
        
        return output_raw_path

    except Exception as e:
        logging.error(f"FAILED to ingest accident data: {str(e)}", exc_info=True)
        raise

# Alias for backwards compatibility
download_accidents = fetch_and_filter_accidents

if __name__ == "__main__":
    fetch_and_filter_accidents()

