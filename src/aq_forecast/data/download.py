"""
download.py — Data acquisition for Hyderabad air quality data.

Fetches historical hourly air quality data from the Open-Meteo Air Quality
API (CAMS reanalysis) and meteorological data from the Open-Meteo Archive
API. Also supports loading pre-downloaded CPCB CSV files.
"""

import os
import logging
import time
import pandas as pd

from src.aq_forecast.aqi import get_aqi_category
import numpy as np
from datetime import datetime, timedelta

from src.aq_forecast.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, POLLUTANT_FEATURES, METEO_FEATURES
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  CPCB Station Metadata for Hyderabad
# ──────────────────────────────────────────────────────────────
HYDERABAD_STATIONS = {
    "Bollaram_Industrial_Area": {
        "id": "site_5029",
        "lat": 17.5400,
        "lon": 78.3500,
        "zone": "Industrial"
    },
    "Central_University_Hyderabad": {
        "id": "site_5085",
        "lat": 17.4600,
        "lon": 78.3300,
        "zone": "Residential"
    },
    "ICRISAT_Patancheru": {
        "id": "site_5086",
        "lat": 17.5100,
        "lon": 78.2700,
        "zone": "Rural"
    },
    "IDA_Pashamylaram": {
        "id": "site_5030",
        "lat": 17.5300,
        "lon": 78.2100,
        "zone": "Industrial"
    },
    "Sanathnagar": {
        "id": "site_5024",
        "lat": 17.4564,
        "lon": 78.4427,
        "zone": "Commercial"
    },
    "Zoo_Park": {
        "id": "site_5025",
        "lat": 17.3500,
        "lon": 78.4510,
        "zone": "Residential"
    },
    "Nacharam": {
        "id": "site_5190",
        "lat": 17.4277,
        "lon": 78.5540,
        "zone": "Industrial"
    },
}


def load_local_csv(file_path: str) -> pd.DataFrame:
    """
    Load air quality data from a local CSV file.

    Expects columns to include datetime-like column and pollutant readings.
    Attempts common column name conventions from CPCB downloads.

    Args:
        file_path: Absolute or relative path to the CSV.

    Returns:
        DataFrame with parsed datetime index.
    """
    logger.info(f"Loading local CSV: {file_path}")

    df = pd.read_csv(file_path)

    # Detect and parse the datetime column
    datetime_candidates = ["Datetime", "datetime", "Date", "date",
                           "From Date", "Sampling Date", "Timestamp"]
    dt_col = None
    for col in datetime_candidates:
        if col in df.columns:
            dt_col = col
            break

    if dt_col is None:
        raise ValueError(
            f"No datetime column found. Available columns: {df.columns.tolist()}"
        )

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    df = df.rename(columns={dt_col: "datetime"})
    df = df.set_index("datetime").sort_index()

    # Clean column names — strip whitespace
    df.columns = df.columns.str.strip()

    # Convert pollutant columns to numeric, coerce errors
    numeric_cols = [c for c in df.columns if c in POLLUTANT_FEATURES + METEO_FEATURES]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    return df


def load_all_station_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load and concatenate CSV files from all stations in the data directory.

    Each CSV should contain data for one monitoring station.
    Files should be named with the station identifier in the filename.

    Args:
        data_dir: Directory containing station CSV files.

    Returns:
        Combined DataFrame with a 'station' column.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    all_frames = []
    csv_files = [f for f in os.listdir(data_dir)
                 if f.endswith(".csv") and f != "all_stations_combined.csv"]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Please download CPCB data and place CSV files there.\n"
            "Download from: https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqm-landing"
        )

    for csv_file in sorted(csv_files):
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = load_local_csv(file_path)
            # Extract station name from filename
            station_name = os.path.splitext(csv_file)[0]
            df["station"] = station_name
            all_frames.append(df)
            logger.info(f"  -> Loaded station '{station_name}': {len(df)} rows")
        except Exception as e:
            logger.warning(f"  -> Skipping {csv_file}: {e}")

    if not all_frames:
        raise ValueError("No valid data loaded from any CSV file.")

    combined = pd.concat(all_frames, axis=0)
    combined = combined.sort_index()
    logger.info(f"Total combined dataset: {len(combined)} rows from "
                f"{len(all_frames)} stations")
    return combined


def fetch_open_meteo_data(lat: float, lon: float,
                          start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical hourly meteorological data from Open-Meteo API.

    This is a free, no-key-required API for historical weather data.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with hourly meteorological readings.
    """
    import urllib.request
    import json

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = (
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,wind_direction_10m,"
        f"precipitation,surface_pressure,"
        f"shortwave_radiation"
        f"&timezone=Asia/Kolkata"
    )

    url = base_url + params
    logger.info(f"Fetching meteorological data from Open-Meteo: {start_date} to {end_date}")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as response:
        data = json.loads(response.read().decode())

    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "AT": hourly["temperature_2m"],
        "RH": hourly["relative_humidity_2m"],
        "WS": hourly["wind_speed_10m"],
        "WD": hourly["wind_direction_10m"],
        "RF": hourly["precipitation"],
        "BP": hourly["surface_pressure"],
        "SR": hourly["shortwave_radiation"],
    })
    df = df.set_index("datetime")

    logger.info(f"Fetched {len(df)} hourly meteorological records")
    return df


def fetch_open_meteo_aq_data(lat: float, lon: float,
                             start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical hourly air quality data from Open-Meteo Air Quality API.

    Data source: CAMS (Copernicus Atmosphere Monitoring Service) global
    reanalysis, which assimilates satellite and ground-station observations.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with hourly pollutant concentrations and datetime index.
    """
    import urllib.request
    import json

    base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = (
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,"
        f"sulphur_dioxide,ozone"
        f"&timezone=Asia/Kolkata"
    )

    url = base_url + params
    logger.info(f"Fetching AQ data from Open-Meteo: {start_date} to {end_date}")

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode())

    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "PM2.5": hourly["pm2_5"],
        "PM10": hourly["pm10"],
        "CO": hourly["carbon_monoxide"],
        "NO2": hourly["nitrogen_dioxide"],
        "SO2": hourly["sulphur_dioxide"],
        "O3": hourly["ozone"],
    })
    # CO from Open-Meteo is in µg/m³; convert to mg/m³ for NAQI breakpoints
    df["CO"] = df["CO"] / 1000.0
    df = df.set_index("datetime")

    logger.info(f"Fetched {len(df)} hourly AQ records")
    return df


def download_all_station_data(start_date: str = "2023-01-01",
                              end_date: str = "2024-12-31",
                              data_dir: str = None) -> pd.DataFrame:
    """
    Download air quality + meteorological data for all Hyderabad stations.

    Fetches data from Open-Meteo APIs (Air Quality + Archive) for each
    CAAQMS station, merges them, computes AQI, and saves per-station CSVs
    to the raw data directory.

    Args:
        start_date: Start of data range (YYYY-MM-DD).
        end_date: End of data range (YYYY-MM-DD).
        data_dir: Directory to save CSVs. Defaults to RAW_DATA_DIR.

    Returns:
        Combined DataFrame with all stations' data.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    all_frames = []
    for station_name, info in HYDERABAD_STATIONS.items():
        lat, lon = info["lat"], info["lon"]
        logger.info(f"Downloading data for {station_name} ({lat}, {lon})...")

        # Fetch air quality data
        aq_df = fetch_open_meteo_aq_data(lat, lon, start_date, end_date)

        # Fetch meteorological data
        meteo_df = fetch_open_meteo_data(lat, lon, start_date, end_date)

        # Merge on datetime index
        merged = aq_df.join(meteo_df, how="left")

        # Compute AQI
        merged["AQI"] = merged.apply(compute_aqi, axis=1)
        merged["AQI_Category"] = merged["AQI"].apply(get_aqi_category)
        merged["station"] = station_name
        merged["zone"] = info["zone"]

        # Save per-station CSV
        csv_path = os.path.join(data_dir, f"{station_name}.csv")
        merged.to_csv(csv_path)
        logger.info(f"  Saved {len(merged)} rows to {csv_path}")

        all_frames.append(merged)

        # Rate-limit: Open-Meteo asks for ≤600 req/min
        time.sleep(1.0)

    combined = pd.concat(all_frames, axis=0).sort_index()
    # Save combined dataset
    combined_path = os.path.join(data_dir, "all_stations_combined.csv")
    combined.to_csv(combined_path)
    logger.info(f"Combined dataset: {len(combined)} rows from "
                f"{len(all_frames)} stations saved to {combined_path}")
    return combined


def merge_meteo_data(air_df: pd.DataFrame,
                     meteo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge meteorological data with air quality data on datetime index.

    Uses nearest-neighbor merge with a 1-hour tolerance.

    Args:
        air_df: Air quality DataFrame with datetime index.
        meteo_df: Meteorological DataFrame with datetime index.

    Returns:
        Merged DataFrame.
    """
    air_df = air_df.copy()
    meteo_df = meteo_df.copy()

    # Reset and merge on datetime as nearest
    air_df = air_df.reset_index()
    meteo_df = meteo_df.reset_index()

    merged = pd.merge_asof(
        air_df.sort_values("datetime"),
        meteo_df.sort_values("datetime"),
        on="datetime",
        tolerance=pd.Timedelta("1h"),
        direction="nearest"
    )
    merged = merged.set_index("datetime")

    logger.info(f"Merged dataset: {len(merged)} rows with "
                f"{len(merged.columns)} columns")
    return merged


def compute_aqi(row: pd.Series) -> float:
    """
    Compute India's National Air Quality Index (NAQI) from sub-index pollutants.

    The AQI is the maximum of individual pollutant sub-indices.
    Uses 24-hour averages for PM2.5, PM10, SO2, NO2, NH3, CO, O3.

    Sub-index breakpoints follow CPCB NAQI standard (2014).

    Args:
        row: A row containing pollutant concentrations.

    Returns:
        Computed AQI value.
    """
    breakpoints = {
        "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
                  (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
        "PM10":  [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
                  (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
        "SO2":   [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
                  (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2100, 401, 500)],
        "NO2":   [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
                  (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500)],
        "CO":    [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
                  (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
        "O3":    [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
                  (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
        "NH3":   [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
                  (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2400, 401, 500)],
    }

    sub_indices = []
    for pollutant, bps in breakpoints.items():
        val = row.get(pollutant, np.nan)
        if pd.isna(val):
            continue
        for c_lo, c_hi, i_lo, i_hi in bps:
            if c_lo <= val <= c_hi:
                sub_index = ((i_hi - i_lo) / (c_hi - c_lo)) * (val - c_lo) + i_lo
                sub_indices.append(sub_index)
                break

    return max(sub_indices) if sub_indices else np.nan


