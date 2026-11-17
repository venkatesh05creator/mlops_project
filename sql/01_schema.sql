-- Schema DDL for Traffic Accident Severity Analytics Pipeline
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS mart;

-- Staging Tables
CREATE TABLE IF NOT EXISTS staging.stg_collision (
    accident_index VARCHAR(50) PRIMARY KEY,
    accident_timestamp TIMESTAMP,
    date_key INT,
    accident_year INT,
    accident_month INT,
    accident_day INT,
    accident_hour INT,
    day_of_week INT,
    day_name VARCHAR(20),
    is_weekend BOOLEAN,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    police_force INT,
    local_authority_district INT,
    accident_severity_code INT,
    accident_severity_label VARCHAR(20),
    road_type_label VARCHAR(50),
    road_surface VARCHAR(50),
    speed_limit INT,
    light_conditions VARCHAR(50),
    weather_conditions VARCHAR(50),
    number_of_vehicles INT,
    number_of_casualties INT
);

CREATE TABLE IF NOT EXISTS staging.stg_weather (
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    timestamp TIMESTAMP,
    temperature_celsius DOUBLE PRECISION,
    precipitation_mm DOUBLE PRECISION,
    wind_speed_kmh DOUBLE PRECISION,
    humidity_pct INT,
    weather_code INT,
    weather_condition VARCHAR(50)
);

-- Analytics Dimension Tables
CREATE TABLE IF NOT EXISTS analytics.dim_severity (
    severity_key INT PRIMARY KEY,
    severity_label VARCHAR(20) NOT NULL
);

CREATE TABLE IF NOT EXISTS analytics.dim_road (
    road_key SERIAL PRIMARY KEY,
    road_type_label VARCHAR(50),
    road_surface VARCHAR(50),
    speed_limit INT,
    light_conditions VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS analytics.dim_weather (
    weather_key SERIAL PRIMARY KEY,
    grid_latitude DOUBLE PRECISION,
    grid_longitude DOUBLE PRECISION,
    weather_timestamp TIMESTAMP,
    temperature_celsius DOUBLE PRECISION,
    precipitation_mm DOUBLE PRECISION,
    wind_speed_kmh DOUBLE PRECISION,
    humidity_pct INT,
    weather_condition VARCHAR(50)
);

-- Analytics Fact Table with PostGIS Spatial Geometry
CREATE TABLE IF NOT EXISTS analytics.fact_accident (
    accident_id SERIAL PRIMARY KEY,
    accident_index VARCHAR(50) UNIQUE NOT NULL,
    accident_timestamp TIMESTAMP NOT NULL,
    date_key INT NOT NULL,
    severity_key INT REFERENCES analytics.dim_severity(severity_key),
    road_key INT REFERENCES analytics.dim_road(road_key),
    weather_key INT REFERENCES analytics.dim_weather(weather_key),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Point, 4326),
    number_of_vehicles INT DEFAULT 1,
    number_of_casualties INT DEFAULT 1,
    grid_lat DOUBLE PRECISION,
    grid_lon DOUBLE PRECISION
);

-- Spatial & Analytical Indexes
CREATE INDEX IF NOT EXISTS idx_fact_accident_geom ON analytics.fact_accident USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_fact_accident_timestamp ON analytics.fact_accident(accident_timestamp);
CREATE INDEX IF NOT EXISTS idx_fact_accident_severity ON analytics.fact_accident(severity_key);

-- Analytical Mart Table for Dashboard Query Performance
CREATE TABLE IF NOT EXISTS mart.accident_summary (
    summary_id SERIAL PRIMARY KEY,
    accident_year INT,
    accident_month INT,
    accident_hour INT,
    day_name VARCHAR(20),
    is_weekend BOOLEAN,
    severity_label VARCHAR(20),
    road_type_label VARCHAR(50),
    weather_condition VARCHAR(50),
    total_accidents INT,
    total_vehicles INT,
    total_casualties INT
);

-- Analytical View for Streamlit Dashboard
CREATE OR REPLACE VIEW mart.vw_accident_analytics AS
SELECT 
    f.accident_index,
    f.accident_timestamp,
    EXTRACT(HOUR FROM f.accident_timestamp)::INT AS accident_hour,
    TRIM(TO_CHAR(f.accident_timestamp, 'Day')) AS day_name,
    f.latitude,
    f.longitude,
    r.road_type_label,
    r.road_surface,
    r.speed_limit,
    r.light_conditions,
    w.weather_condition,
    w.temperature_celsius,
    w.precipitation_mm,
    s.severity_label AS accident_severity_label,
    f.number_of_vehicles,
    f.number_of_casualties
FROM analytics.fact_accident f
JOIN analytics.dim_severity s ON f.severity_key = s.severity_key
JOIN analytics.dim_road r ON f.road_key = r.road_key
LEFT JOIN analytics.dim_weather w ON f.weather_key = w.weather_key;

