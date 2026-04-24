from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import requests
from requests import RequestException

from src.data.cities import City

OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY_FIELDS: tuple[str, ...] = (
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
)


def build_params(city: City, start_date: str, end_date: str) -> dict[str, str | float]:
    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": ",".join(HOURLY_FIELDS),
    }


def fetch_city_history(city: City, start_date: str, end_date: str, timeout: int) -> pd.DataFrame:
    params = build_params(city, start_date, end_date)
    response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise ValueError(f"No hourly data returned for {city.name}")

    data: dict[str, Iterable[object]] = {"timestamp": times}
    for field in HOURLY_FIELDS:
        values = hourly.get(field, [None] * len(times))
        if len(values) != len(times):
            raise ValueError(
                f"Length mismatch for {city.name}: field={field}, time={len(times)}, values={len(values)}"
            )
        data[field] = values

    df = pd.DataFrame(data)
    df.insert(0, "city", city.name)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    nat_count = int(df["timestamp"].isna().sum())
    if nat_count > 0:
        raise ValueError(f"Invalid timestamps for {city.name}: NaT rows={nat_count}")
    return df


def fetch_latest_city_observation(city: City, timeout: int, lookback_days: int = 2) -> dict[str, object]:
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback_days)
    frame = fetch_city_history(city, start_dt.isoformat(), end_dt.isoformat(), timeout=timeout)
    latest = frame.sort_values("timestamp").tail(1).iloc[0].to_dict()
    return latest