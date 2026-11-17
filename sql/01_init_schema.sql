-- PostgreSQL / PostGIS Schema Initialization Script
-- Traffic Accident Severity Analytics

CREATE EXTENSION IF NOT EXISTS postgis;

-- Create Schema Layers
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS dw;
CREATE SCHEMA IF NOT EXISTS mart;

-- ============================================================================
-- DIMENSIONAL WAREHOUSE LAYER (dw)
-- ============================================================================

-- 1. Date Dimension
CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key INT PRIMARY KEY,              -- YYYYMMDD
    full_date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    month_name VARCHAR(20) NOT NULL,
    day_of_month INT NOT NULL,
    day_of_week INT NOT NULL,              -- 1=Sunday..7=Saturday or 1=Monday..7=Sunday
    day_name VARCHAR(20) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    quarter INT NOT NULL
);

-- 2. Location Dimension (PostGIS spatial Point)
CREATE TABLE IF NOT EXISTS dw.dim_location (
    location_key SERIAL PRIMARY KEY,
    latitude NUMERIC(9,6) NOT NULL,
    longitude NUMERIC(9,6) NOT NULL,
    police_force INT,
    police_force_name VARCHAR(100),
    local_authority_district INT,
    local_authority_name VARCHAR(100),
    geom GEOMETRY(Point, 4326)
);

CREATE INDEX IF NOT EXISTS idx_dim_location_geom ON dw.dim_location USING GIST (geom);

-- 3. Road Dimension
CREATE TABLE IF NOT EXISTS dw.dim_road (
    road_key SERIAL PRIMARY KEY,
    road_class VARCHAR(50),
    road_surface VARCHAR(50),
    speed_limit INT,
    junction_detail VARCHAR(100),
    light_conditions VARCHAR(100),
    urban_or_rural VARCHAR(20)
);

-- 4. Weather Dimension
CREATE TABLE IF NOT EXISTS dw.dim_weather (
    weather_key SERIAL PRIMARY KEY,
    weather_code INT,
    weather_condition VARCHAR(100),
    temperature_celsius NUMERIC(4,1),
    precipitation_mm NUMERIC(5,2),
    wind_speed_kmh NUMERIC(5,2),
    humidity_pct INT,
    weather_severity_index VARCHAR(20) -- Clear, Mild, Moderate, Severe
);

-- 5. Severity & Incident Classification Dimension
CREATE TABLE IF NOT EXISTS dw.dim_severity (
    severity_key SERIAL PRIMARY KEY,
    accident_severity_code INT NOT NULL,
    accident_severity_label VARCHAR(30) NOT NULL, -- Slight, Serious, Fatal
    number_of_vehicles INT NOT NULL,
    number_of_casualties INT NOT NULL
);

-- 6. Fact Accidents Table
CREATE TABLE IF NOT EXISTS dw.fact_accidents (
    accident_index VARCHAR(50) PRIMARY KEY,
    date_key INT NOT NULL REFERENCES dw.dim_date(date_key),
    location_key INT NOT NULL REFERENCES dw.dim_location(location_key),
    road_key INT NOT NULL REFERENCES dw.dim_road(road_key),
    weather_key INT REFERENCES dw.dim_weather(weather_key),
    severity_key INT NOT NULL REFERENCES dw.dim_severity(severity_key),
    accident_timestamp TIMESTAMP NOT NULL,
    accident_hour INT NOT NULL,
    casualty_count INT NOT NULL,
    vehicle_count INT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_date ON dw.fact_accidents(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_location ON dw.fact_accidents(location_key);
CREATE INDEX IF NOT EXISTS idx_fact_road ON dw.fact_accidents(road_key);
CREATE INDEX IF NOT EXISTS idx_fact_weather ON dw.fact_accidents(weather_key);
CREATE INDEX IF NOT EXISTS idx_fact_severity ON dw.fact_accidents(severity_key);

-- ============================================================================
-- ANALYTICAL MART LAYER (mart)
-- ============================================================================

CREATE TABLE IF NOT EXISTS mart.monthly_hourly_accident_summary (
    summary_id SERIAL PRIMARY KEY,
    year INT NOT NULL,
    month INT NOT NULL,
    day_of_week INT NOT NULL,
    day_name VARCHAR(20) NOT NULL,
    accident_hour INT NOT NULL,
    road_surface VARCHAR(50),
    weather_condition VARCHAR(100),
    accident_severity_label VARCHAR(30),
    total_accidents INT NOT NULL,
    total_casualties INT NOT NULL,
    total_vehicles INT NOT NULL
);

-- View for Dashboard Aggregations
CREATE OR REPLACE VIEW mart.vw_accident_analytics AS
SELECT 
    f.accident_index,
    f.accident_timestamp,
    f.accident_hour,
    d.full_date,
    d.year,
    d.month,
    d.month_name,
    d.day_name,
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
FROM dw.fact_accidents f
JOIN dw.dim_date d ON f.date_key = d.date_key
JOIN dw.dim_location l ON f.location_key = l.location_key
JOIN dw.dim_road r ON f.road_key = r.road_key
LEFT JOIN dw.dim_weather w ON f.weather_key = w.weather_key
JOIN dw.dim_severity s ON f.severity_key = s.severity_key;
