"""
Database migrations and maintenance for air quality platform.

This script:
1. Creates materialized views for hourly/daily/monthly aggregates
2. Creates forecasts tracking table
3. Sets up refresh jobs
4. Adds indexes for performance
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# SQL migrations

MIGRATION_1_HOURLY_AGGREGATES = """
-- Hourly aggregates (computed every hour)
CREATE MATERIALIZED VIEW IF NOT EXISTS aq.hourly_aggregates AS
SELECT
    city_slug,
    city_name,
    DATE_TRUNC('hour', observed_at) as hour,
    COUNT(*) as obs_count,
    AVG(pm2_5) as pm2_5_avg,
    MAX(pm2_5) as pm2_5_max,
    MIN(pm2_5) as pm2_5_min,
    STDDEV(pm2_5) as pm2_5_stddev,
    AVG(pm10) as pm10_avg,
    MAX(pm10) as pm10_max,
    MIN(pm10) as pm10_min,
    STDDEV(pm10) as pm10_stddev,
    AVG(carbon_monoxide) as co_avg,
    MAX(carbon_monoxide) as co_max,
    MIN(carbon_monoxide) as co_min,
    AVG(nitrogen_dioxide) as no2_avg,
    MAX(nitrogen_dioxide) as no2_max,
    MIN(nitrogen_dioxide) as no2_min,
    AVG(sulphur_dioxide) as so2_avg,
    MAX(sulphur_dioxide) as so2_max,
    MIN(sulphur_dioxide) as so2_min,
    AVG(ozone) as o3_avg,
    MAX(ozone) as o3_max,
    MIN(ozone) as o3_min,
    AVG(us_aqi) as us_aqi_avg,
    MAX(us_aqi) as us_aqi_max,
    MIN(us_aqi) as us_aqi_min,
    NOW() as computed_at
FROM aq.observations
WHERE observed_at >= NOW() - INTERVAL '90 days'
GROUP BY city_slug, city_name, DATE_TRUNC('hour', observed_at)
ORDER BY city_slug, hour DESC;

CREATE INDEX IF NOT EXISTS idx_hourly_agg_city_hour 
    ON aq.hourly_aggregates (city_slug, hour DESC);
"""

MIGRATION_2_DAILY_AGGREGATES = """
-- Daily aggregates (computed every day)
CREATE MATERIALIZED VIEW IF NOT EXISTS aq.daily_aggregates AS
SELECT
    city_slug,
    city_name,
    DATE_TRUNC('day', observed_at)::DATE as day,
    COUNT(*) as obs_count,
    AVG(pm2_5) as pm2_5_avg,
    MAX(pm2_5) as pm2_5_max,
    MIN(pm2_5) as pm2_5_min,
    STDDEV(pm2_5) as pm2_5_stddev,
    AVG(pm10) as pm10_avg,
    MAX(pm10) as pm10_max,
    MIN(pm10) as pm10_min,
    STDDEV(pm10) as pm10_stddev,
    AVG(carbon_monoxide) as co_avg,
    MAX(carbon_monoxide) as co_max,
    MIN(carbon_monoxide) as co_min,
    AVG(nitrogen_dioxide) as no2_avg,
    MAX(nitrogen_dioxide) as no2_max,
    MIN(nitrogen_dioxide) as no2_min,
    AVG(sulphur_dioxide) as so2_avg,
    MAX(sulphur_dioxide) as so2_max,
    MIN(sulphur_dioxide) as so2_min,
    AVG(ozone) as o3_avg,
    MAX(ozone) as o3_max,
    MIN(ozone) as o3_min,
    AVG(us_aqi) as us_aqi_avg,
    MAX(us_aqi) as us_aqi_max,
    MIN(us_aqi) as us_aqi_min,
    NOW() as computed_at
FROM aq.observations
WHERE observed_at >= NOW() - INTERVAL '365 days'
GROUP BY city_slug, city_name, DATE_TRUNC('day', observed_at)
ORDER BY city_slug, day DESC;

CREATE INDEX IF NOT EXISTS idx_daily_agg_city_day 
    ON aq.daily_aggregates (city_slug, day DESC);
"""

MIGRATION_3_MONTHLY_AGGREGATES = """
-- Monthly aggregates (computed monthly)
CREATE MATERIALIZED VIEW IF NOT EXISTS aq.monthly_aggregates AS
SELECT
    city_slug,
    city_name,
    DATE_TRUNC('month', observed_at)::DATE as month,
    COUNT(*) as obs_count,
    AVG(pm2_5) as pm2_5_avg,
    MAX(pm2_5) as pm2_5_max,
    MIN(pm2_5) as pm2_5_min,
    AVG(pm10) as pm10_avg,
    MAX(pm10) as pm10_max,
    MIN(pm10) as pm10_min,
    AVG(carbon_monoxide) as co_avg,
    MAX(carbon_monoxide) as co_max,
    MIN(carbon_monoxide) as co_min,
    AVG(nitrogen_dioxide) as no2_avg,
    MAX(nitrogen_dioxide) as no2_max,
    MIN(nitrogen_dioxide) as no2_min,
    AVG(sulphur_dioxide) as so2_avg,
    MAX(sulphur_dioxide) as so2_max,
    MIN(sulphur_dioxide) as so2_min,
    AVG(ozone) as o3_avg,
    MAX(ozone) as o3_max,
    MIN(ozone) as o3_min,
    AVG(us_aqi) as us_aqi_avg,
    MAX(us_aqi) as us_aqi_max,
    MIN(us_aqi) as us_aqi_min,
    NOW() as computed_at
FROM aq.observations
WHERE observed_at >= NOW() - INTERVAL '2 years'
GROUP BY city_slug, city_name, DATE_TRUNC('month', observed_at)
ORDER BY city_slug, month DESC;

CREATE INDEX IF NOT EXISTS idx_monthly_agg_city_month 
    ON aq.monthly_aggregates (city_slug, month DESC);
"""

MIGRATION_4_FORECASTS_TABLE = """
-- Track all forecasts for accuracy analytics
CREATE TABLE IF NOT EXISTS aq.forecasts (
    forecast_id TEXT PRIMARY KEY,
    city_slug TEXT NOT NULL REFERENCES aq.cities(city_slug) ON DELETE CASCADE,
    city_name TEXT NOT NULL,
    forecast_timestamp TIMESTAMPTZ NOT NULL,
    horizon_timestamp TIMESTAMPTZ NOT NULL,
    horizon_hours INT NOT NULL,
    predicted_us_aqi DOUBLE PRECISION NOT NULL,
    predicted_pm2_5 DOUBLE PRECISION,
    predicted_pm10 DOUBLE PRECISION,
    model_type TEXT NOT NULL,
    model_version TEXT,
    confidence DOUBLE PRECISION,
    actual_us_aqi DOUBLE PRECISION,
    actual_pm2_5 DOUBLE PRECISION,
    actual_pm10 DOUBLE PRECISION,
    prediction_error DOUBLE PRECISION,
    absolute_error DOUBLE PRECISION,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_forecasts_city_horizon 
    ON aq.forecasts (city_slug, horizon_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_forecasts_verification 
    ON aq.forecasts (verified_at DESC) WHERE verified_at IS NOT NULL;

-- View for forecast accuracy tracking
CREATE OR REPLACE VIEW aq.forecast_accuracy AS
SELECT
    city_slug,
    city_name,
    model_type,
    COUNT(*) as total_predictions,
    COUNT(verified_at) as verified_count,
    ROUND(AVG(ABS(prediction_error))::NUMERIC, 2) as mean_absolute_error,
    ROUND(STDDEV(ABS(prediction_error))::NUMERIC, 2) as stddev_error,
    ROUND((
        100.0 * STDDEV(ABS(prediction_error)) / NULLIF(AVG(ABS(prediction_error)), 0)
    )::NUMERIC, 2) as error_coefficient_variation,
    MAX(verified_at) as last_verified
FROM aq.forecasts
WHERE verified_at IS NOT NULL
GROUP BY city_slug, city_name, model_type
ORDER BY city_slug, model_type;
"""

MIGRATION_5_LATEST_SNAPSHOT = """
-- View for latest observation per city (materialized for performance)
CREATE MATERIALIZED VIEW IF NOT EXISTS aq.latest_observations AS
SELECT DISTINCT ON (city_slug)
    city_slug,
    city_name,
    observed_at,
    pm2_5,
    pm10,
    carbon_monoxide,
    nitrogen_dioxide,
    sulphur_dioxide,
    ozone,
    us_aqi,
    source,
    ingested_at,
    created_at
FROM aq.observations
WHERE observed_at >= NOW() - INTERVAL '7 days'
ORDER BY city_slug, observed_at DESC;

CREATE INDEX IF NOT EXISTS idx_latest_obs_city 
    ON aq.latest_observations (city_slug);
"""

MIGRATION_6_CLEANUP_POLICIES = """
-- Mark old observations for archive (keep 365 days, mark others for deletion)
CREATE TABLE IF NOT EXISTS aq.archive_metadata (
    observation_id BIGSERIAL PRIMARY KEY,
    city_slug TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    archive_date TIMESTAMPTZ,
    archived_to_gzip TEXT,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ,
    UNIQUE (city_slug, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_archive_city_date 
    ON aq.archive_metadata (city_slug, observed_at DESC);
"""

def run_migrations(connection):
    """Execute all migrations."""
    cursor = connection.cursor()
    
    migrations = [
        ("Hourly aggregates", MIGRATION_1_HOURLY_AGGREGATES),
        ("Daily aggregates", MIGRATION_2_DAILY_AGGREGATES),
        ("Monthly aggregates", MIGRATION_3_MONTHLY_AGGREGATES),
        ("Forecasts table", MIGRATION_4_FORECASTS_TABLE),
        ("Latest snapshot view", MIGRATION_5_LATEST_SNAPSHOT),
        ("Cleanup policies", MIGRATION_6_CLEANUP_POLICIES),
    ]
    
    for name, sql in migrations:
        try:
            cursor.execute(sql)
            connection.commit()
            print(f"OK {name}")
        except Exception as e:
            connection.rollback()
            print(f"FAILED {name}: {e}")
            raise

def refresh_materialized_views(connection):
    """Refresh all materialized views."""
    cursor = connection.cursor()
    
    views = [
        "aq.hourly_aggregates",
        "aq.daily_aggregates",
        "aq.monthly_aggregates",
        "aq.latest_observations",
    ]
    
    for view_name in views:
        try:
            cursor.execute(f"REFRESH MATERIALIZED VIEW {view_name}")
            connection.commit()
            print(f"OK Refreshed {view_name}")
        except Exception as e:
            connection.rollback()
            print(f"FAILED Refreshed {view_name}: {e}")
            raise

if __name__ == "__main__":
    from src.ingestion.india_aq import IngestionSettings, connect, ensure_schema
    
    settings = IngestionSettings.from_env()
    
    try:
        conn = connect(settings.dsn)
        print("Connected to PostgreSQL")

        ensure_schema(conn, settings.schema_name)
        
        print("\n=== Running Migrations ===")
        run_migrations(conn)
        
        print("\n=== Refreshing Materialized Views ===")
        refresh_materialized_views(conn)
        
        conn.close()
        print("\nOK Database setup complete!")
        
    except Exception as e:
        print(f"FAILED Error: {e}")
        sys.exit(1)
