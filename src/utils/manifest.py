import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
# pyrefly: ignore [missing-import]
import config

MANIFEST_DIR = Path(config.LOGS_DIR) / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

REJECTION_LOG_PATH = Path(config.LOGS_DIR) / "rejections.csv"

def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def record_manifest(task_name: str, source_url: str, output_file: Path, row_count: int, metadata: Optional[dict] = None) -> dict:
    """Creates and persists a run manifest for pipeline tasks."""
    output_path = Path(output_file)
    manifest = {
        "task_name": task_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "output_file": str(output_path),
        "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "sha256_hash": compute_file_sha256(output_path) if output_path.exists() else None,
        "row_count": row_count,
        "metadata": metadata or {}
    }
    
    manifest_file = MANIFEST_DIR / f"{task_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    return manifest

def log_rejections(rejected_df: pd.DataFrame, reason: str):
    """Appends rejected rows to the rejections audit log."""
    if rejected_df.empty:
        return
        
    rejected_copy = rejected_df.copy()
    rejected_copy["rejection_reason"] = reason
    rejected_copy["rejected_at"] = datetime.now(timezone.utc).isoformat()
    
    file_exists = REJECTION_LOG_PATH.exists()
    rejected_copy.to_csv(REJECTION_LOG_PATH, mode='a', header=not file_exists, index=False)
