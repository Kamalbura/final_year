from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import pandas as pd
import requests
from requests import RequestException

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.cities import City, INDIA_MAJOR_CITIES

OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY_FIELDS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]


def parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("timeout must be a positive integer")
        return parsed

    parser = argparse.ArgumentParser(
        description="Download 1-year hourly air-quality data for major Indian cities."
    )
    parser.add_argument(
        "--output-dir",
        default="data/india_aq_1y",
        help="Directory to store per-city and combined CSV files.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date in YYYY-MM-DD format. Defaults to today - 365 days.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_int,
        default=45,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def resolve_dates(start_date_raw: str | None, end_date_raw: str | None) -> tuple[str, str]:
    end_dt = date.fromisoformat(end_date_raw) if end_date_raw else date.today()
    start_dt = date.fromisoformat(start_date_raw) if start_date_raw else end_dt - timedelta(days=365)
    if start_dt > end_dt:
        raise ValueError("start-date must be earlier than or equal to end-date")
    return start_dt.isoformat(), end_dt.isoformat()


def build_params(city: City, start_date: str, end_date: str) -> Dict[str, str | float]:
    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        "hourly": ",".join(HOURLY_FIELDS),
    }


def fetch_city(city: City, start_date: str, end_date: str, timeout: int, retries: int = 3) -> pd.DataFrame:
    params = build_params(city, start_date, end_date)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            hourly = payload.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                raise ValueError(f"No hourly data returned for {city.name}")

            data = {"timestamp": times}
            for field in HOURLY_FIELDS:
                values = hourly.get(field, [None] * len(times))
                if len(values) != len(times):
                    raise ValueError(
                        f"Length mismatch for {city.name}: field={field}, "
                        f"time={len(times)}, values={len(values)}"
                    )
                data[field] = values

            df = pd.DataFrame(data)
            df.insert(0, "city", city.name)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            nat_count = int(df["timestamp"].isna().sum())
            if nat_count > 0:
                raise ValueError(f"Invalid timestamps for {city.name}: NaT rows={nat_count}")
            return df
        except RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            continue
        except ValueError:
            raise
    raise RuntimeError(f"Failed to download {city.name} after {retries} attempts: {last_error}")


def save_city_csv(df: pd.DataFrame, output_dir: Path, city_name: str) -> Path:
    file_name = city_name.lower().replace(" ", "_") + "_aq_1y.csv"
    output_path = output_dir / file_name
    df.to_csv(output_path, index=False)
    return output_path


def download_all_cities(
    cities: Iterable[City], start_date: str, end_date: str, output_dir: Path, timeout: int
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    failures: List[str] = []

    for city in cities:
        print(f"Downloading {city.name} ({start_date} to {end_date})...")
        try:
            city_df = fetch_city(city, start_date, end_date, timeout=timeout)
            csv_path = save_city_csv(city_df, output_dir, city.name)
            frames.append(city_df)
            print(f"Saved: {csv_path} | rows={len(city_df)}")
        except (RuntimeError, ValueError) as exc:
            failures.append(f"{city.name}: {exc}")
            print(f"Failed: {city.name} | {exc}")

    if not frames:
        failure_report = "\n".join(failures) if failures else "unknown failure"
        raise RuntimeError(f"No city data was downloaded successfully.\n{failure_report}")

    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "india_major_cities_aq_1y_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined dataset: {combined_path} | rows={len(combined)}")

    if failures:
        failed_path = output_dir / "failed_cities.log"
        failed_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"Some cities failed. See: {failed_path}")

    return combined


def main() -> None:
    try:
        args = parse_args()
        start_date, end_date = resolve_dates(args.start_date, args.end_date)
        output_dir = Path(args.output_dir)
        combined = download_all_cities(
            cities=INDIA_MAJOR_CITIES,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            timeout=args.timeout,
        )

        summary = combined.groupby("city").size().rename("row_count").reset_index()
        summary_path = output_dir / "dataset_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
    except ValueError as exc:
        raise SystemExit(f"Input validation error: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
