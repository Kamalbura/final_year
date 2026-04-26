-- Migration 1: Hourly aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS aq.hourly_aggregates AS
SELECT
    city_slug,
    city_name,
    DATE_TRUNC('hour', observed_at) as hour,
    COUNT(*) as obs_count,
    AVG(pm2_5) as pm2_5_avg, MAX(pm2_5) as pm2_5_max, MIN(pm2_5) as pm2_5_min,
    AVG(pm10) as pm10_avg, MAX(pm10) as pm10_max, MIN(pm10) as pm10_min,
    AVG(carbon_monoxide) as co_avg, MAX(carbon_monoxide) as co_max, MIN(carbon_monoxide) as co_min,
    AVG(nitrogen_dioxide) as no2_avg, MAX(nitrogen_dioxide) as no2_max, MIN(nitrogen_dioxide) as no2_min,
    AVG(sulphur_dioxide) as so2_avg, MAX(sulphur_dioxide) as so2_max, MIN(sulphur_dioxide) as so2_min,
    AVG(ozone) as o3_avg, MAX(ozone) as o3_max, MIN(ozone) as o3_min,
    AVG(us_aqi) as us_aqi_avg, MAX(us_aqi) as us_aqi_max, MIN(us_aqi) as us_aqi_min,
    NOW() as computed_at
FROM aq.observations
WHERE observed_at >= NOW() - INTERVAL '90 days'
GROUP BY city_slug, city_name, DATE_TRUNC('hour', observed_at)
ORDER BY city_slug, hour DESC;

CREATE INDEX IF NOT EXISTS idx_hourly_agg_city_hour 
    ON aq.hourly_aggregates (city_slug, hour DESC);

-- Migration 2: Daily aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS aq.daily_aggregates AS
SELECT
    city_slug,
    city_name,
    DATE_TRUNC('day', observed_at)::DATE as day,
    COUNT(*) as obs_count,
    AVG(pm2_5) as pm2_5_avg, MAX(pm2_5) as pm2_5_max, MIN(pm2_5) as pm2_5_min,
    AVG(pm10) as pm10_avg, MAX(pm10) as pm10_max, MIN(pm10) as pm10_min,
    AVG(carbon_monoxide) as co_avg, MAX(carbon_monoxide) as co_max, MIN(carbon_monoxide) as co_min,
    AVG(nitrogen_dioxide) as no2_avg, MAX(nitrogen_dioxide) as no2_max, MIN(nitrogen_dioxide) as no2_min,
    AVG(sulphur_dioxide) as so2_avg, MAX(sulphur_dioxide) as so2_max, MIN(sulphur_dioxide) as so2_min,
    AVG(ozone) as o3_avg, MAX(ozone) as o3_max, MIN(ozone) as o3_min,
    AVG(us_aqi) as us_aqi_avg, MAX(us_aqi) as us_aqi_max, MIN(us_aqi) as us_aqi_min,
    NOW() as computed_at
FROM aq.observations
WHERE observed_at >= NOW() - INTERVAL '365 days'
GROUP BY city_slug, city_name, DATE_TRUNC('day', observed_at)
ORDER BY city_slug, day DESC;

CREATE INDEX IF NOT EXISTS idx_daily_agg_city_day 
    ON aq.daily_aggregates (city_slug, day DESC);

-- Migration 3: Monthly aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS aq.monthly_aggregates AS
SELECT
    city_slug,
    city_name,
    DATE_TRUNC('month', observed_at)::DATE as month,
    COUNT(*) as obs_count,
    AVG(pm2_5) as pm2_5_avg, MAX(pm2_5) as pm2_5_max, MIN(pm2_5) as pm2_5_min,
    AVG(pm10) as pm10_avg, MAX(pm10) as pm10_max, MIN(pm10) as pm10_min,
    AVG(carbon_monoxide) as co_avg, MAX(carbon_monoxide) as co_max, MIN(carbon_monoxide) as co_min,
    AVG(nitrogen_dioxide) as no2_avg, MAX(nitrogen_dioxide) as no2_max, MIN(nitrogen_dioxide) as no2_min,
    AVG(sulphur_dioxide) as so2_avg, MAX(sulphur_dioxide) as so2_max, MIN(sulphur_dioxide) as so2_min,
    AVG(ozone) as o3_avg, MAX(ozone) as o3_max, MIN(ozone) as o3_min,
    AVG(us_aqi) as us_aqi_avg, MAX(us_aqi) as us_aqi_max, MIN(us_aqi) as us_aqi_min,
    NOW() as computed_at
FROM aq.observations
WHERE observed_at >= NOW() - INTERVAL '2 years'
GROUP BY city_slug, city_name, DATE_TRUNC('month', observed_at)
ORDER BY city_slug, month DESC;

CREATE INDEX IF NOT EXISTS idx_monthly_agg_city_month 
    ON aq.monthly_aggregates (city_slug, month DESC);

-- Migration 4: Forecasts table
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

-- Migration 5: Latest observation snapshot
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

-- Migration 6: Archive metadata
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
