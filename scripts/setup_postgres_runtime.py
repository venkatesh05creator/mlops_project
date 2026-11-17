import os
import sys
import shutil
import zipfile
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
DOWNLOADS_DIR = os.path.join(RUNTIME_DIR, "downloads")
PG_INSTALL_DIR = os.path.join(RUNTIME_DIR, "pgsql")
PGDATA_DIR = os.path.join(RUNTIME_DIR, "pgdata")
LOGS_DIR = os.path.join(RUNTIME_DIR, "logs")

PG_ZIP = os.path.join(DOWNLOADS_DIR, "pg16-binaries.zip")
POSTGIS_ZIP = os.path.join(DOWNLOADS_DIR, "postgis-bundle.zip")

def log(msg):
    print(f"[POSTGRES RUNTIME SETUP] {msg}", flush=True)

def main():
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # 1. Extract PostgreSQL Binaries if missing or incomplete
    bki_file = os.path.join(PG_INSTALL_DIR, "share", "postgres.bki")
    if not os.path.exists(bki_file):
        log(f"Extracting PostgreSQL 16 binaries from {PG_ZIP} into {RUNTIME_DIR}...")
        with zipfile.ZipFile(PG_ZIP, 'r') as zip_ref:
            zip_ref.extractall(RUNTIME_DIR)
        log("PostgreSQL binaries extracted successfully.")
    else:
        log(f"PostgreSQL installation verified at {PG_INSTALL_DIR}.")

    # 2. Extract PostGIS bundle into PG_INSTALL_DIR
    postgis_control = os.path.join(PG_INSTALL_DIR, "share", "extension", "postgis.control")
    if not os.path.exists(postgis_control) and os.path.exists(POSTGIS_ZIP):
        log(f"Extracting PostGIS bundle overlay from {POSTGIS_ZIP} into {PG_INSTALL_DIR}...")
        with zipfile.ZipFile(POSTGIS_ZIP, 'r') as zip_ref:
            for member in zip_ref.infolist():
                # Strip leading top directory name from postgis zip (e.g., 'postgis-bundle-pg16-3.6.2x64/')
                parts = member.filename.split('/', 1)
                if len(parts) > 1 and parts[1]:
                    target_path = os.path.join(PG_INSTALL_DIR, parts[1])
                    if member.is_dir():
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
        log("PostGIS overlay extracted successfully.")

    bin_dir = os.path.join(PG_INSTALL_DIR, "bin")
    initdb_exe = os.path.join(bin_dir, "initdb.exe")
    pg_ctl_exe = os.path.join(bin_dir, "pg_ctl.exe")
    psql_exe = os.path.join(bin_dir, "psql.exe")

    os.environ["PATH"] = bin_dir + os.path.pathsep + os.environ.get("PATH", "")

    # 3. Init DB cluster if not initialized
    if not os.path.exists(os.path.join(PGDATA_DIR, "PG_VERSION")):
        log(f"Initializing Postgres cluster in {PGDATA_DIR}...")
        cmd = [initdb_exe, "-D", PGDATA_DIR, "-U", "postgres", "-A", "trust", "--encoding=UTF8", "-L", os.path.join(PG_INSTALL_DIR, "share")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        log(f"initdb stdout: {res.stdout}")
        if res.returncode != 0:
            log(f"initdb stderr: {res.stderr}")
            sys.exit(1)
        
        conf_file = os.path.join(PGDATA_DIR, "postgresql.conf")
        if os.path.exists(conf_file):
            with open(conf_file, "a") as f:
                f.write("\nport = 5433\nlisten_addresses = 'localhost'\n")

    # 4. Start PostgreSQL Server
    log("Checking PostgreSQL server status...")
    status_cmd = [pg_ctl_exe, "-D", PGDATA_DIR, "status"]
    status_res = subprocess.run(status_cmd, capture_output=True, text=True)
    if "is running" not in status_res.stdout:
        log("Starting PostgreSQL server on port 5433...")
        log_file = os.path.join(LOGS_DIR, "postgres.log")
        start_cmd = [pg_ctl_exe, "-D", PGDATA_DIR, "-l", log_file, "-o", "-p 5433", "start"]
        res = subprocess.run(start_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        status_check = subprocess.run(status_cmd, capture_output=True, text=True)
        log(f"pg_ctl status after start: {status_check.stdout.strip()}")

    # 5. Create database 'traffic' and enable postgis
    log("Creating database 'traffic' if not exists...")
    check_db_cmd = [psql_exe, "-U", "postgres", "-p", "5433", "-h", "localhost", "-d", "postgres", "-tc", "SELECT 1 FROM pg_database WHERE datname = 'traffic'"]
    res = subprocess.run(check_db_cmd, capture_output=True, text=True)
    if "1" not in res.stdout:
        create_db_cmd = [psql_exe, "-U", "postgres", "-p", "5433", "-h", "localhost", "-d", "postgres", "-c", "CREATE DATABASE traffic;"]
        subprocess.run(create_db_cmd, check=True)
        log("Database 'traffic' created.")

    log("Enabling PostGIS extension in 'traffic' database...")
    ext_cmd = [psql_exe, "-U", "postgres", "-p", "5433", "-h", "localhost", "-d", "traffic", "-c", "CREATE EXTENSION IF NOT EXISTS postgis;"]
    ext_res = subprocess.run(ext_cmd, capture_output=True, text=True)
    log(f"PostGIS extension output: {ext_res.stdout}")
    if ext_res.returncode != 0:
        log(f"PostGIS extension stderr: {ext_res.stderr}")
    else:
        log("SUCCESS: PostgreSQL 16 + PostGIS 3.6 runtime is UP and RUNNING on localhost:5433/traffic!")

if __name__ == "__main__":
    main()
