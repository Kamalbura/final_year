CREATE SCHEMA IF NOT EXISTS aq;

CREATE TABLE IF NOT EXISTS aq.cities (
    city_slug TEXT PRIMARY KEY,
    city_name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS aq.city_watermarks (
    city_slug TEXT PRIMARY KEY REFERENCES aq.cities(city_slug) ON DELETE CASCADE,
    last_observed_at TIMESTAMPTZ,
    overlap_hours INTEGER NOT NULL DEFAULT 6,
    source TEXT NOT NULL DEFAULT 'open-meteo',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS aq.ingestion_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    cities_total INTEGER NOT NULL DEFAULT 0,
    cities_succeeded INTEGER NOT NULL DEFAULT 0,
    cities_failed INTEGER NOT NULL DEFAULT 0,
    rows_fetched INTEGER NOT NULL DEFAULT 0,
    rows_upserted INTEGER NOT NULL DEFAULT 0,
    archive_root TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS aq.observations (
    city_slug TEXT NOT NULL REFERENCES aq.cities(city_slug) ON DELETE CASCADE,
    city_name TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    pm2_5 DOUBLE PRECISION,
    pm10 DOUBLE PRECISION,
    carbon_monoxide DOUBLE PRECISION,
    nitrogen_dioxide DOUBLE PRECISION,
    sulphur_dioxide DOUBLE PRECISION,
    ozone DOUBLE PRECISION,
    us_aqi DOUBLE PRECISION,
    source TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    run_id TEXT NOT NULL REFERENCES aq.ingestion_runs(run_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (city_slug, observed_at, source)
);

CREATE INDEX IF NOT EXISTS idx_observations_city_time_desc
    ON aq.observations (city_slug, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_time_desc
    ON aq.observations (observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_observations_run_id
    ON aq.observations (run_id);
