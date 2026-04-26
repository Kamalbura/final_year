from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CKAN_PACKAGE_URL = "https://data.opencity.in/api/3/action/package_show?id=hyderabad-hourly-air-quality-reports"
OUTPUT_DIR = Path("data/hyderabad_station_aq_1y")


@dataclass(frozen=True)
class StationTarget:
    canonical_name: str
    aliases: tuple[str, ...]


TARGET_STATIONS: tuple[StationTarget, ...] = (
    StationTarget("ICRISAT Patancheru", ("ICRISAT Patancheru", "Patancheru")),
    StationTarget("Central University (UoH)", ("Central University", "UoH", "University of Hyderabad")),
    StationTarget("IITH Kandi", ("IITH Kandi", "IIIT", "Kandi")),
)

REQUIRED_METRICS = ("pm2_5", "pm10")


def create_retry_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET",)),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def load_dataset_metadata(session: requests.Session) -> dict:
    response = session.get(CKAN_PACKAGE_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError("CKAN package_show returned unsuccessful response")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Invalid CKAN payload shape")
    return result


def iter_station_resources(resources: Iterable[dict]) -> Iterable[dict]:
    for resource in resources:
        name = str(resource.get("name", ""))
        fmt = str(resource.get("format", "")).upper()
        url = str(resource.get("url", ""))
        if fmt != "CSV" or not url:
            continue
        if "15 minute AQI Data for 2024-25" in name:
            yield resource


def station_match_score(name: str, aliases: tuple[str, ...]) -> int:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    score = 0
    for alias in aliases:
        alias_norm = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
        if not alias_norm:
            continue
        if normalized == alias_norm:
            score = max(score, 100)
        elif re.search(rf"\b{re.escape(alias_norm)}\b", normalized):
            score = max(score, 80)
        elif alias_norm in normalized:
            score = max(score, 50)
    return score


def find_resource_for_target(resources: list[dict], target: StationTarget) -> dict:
    scored: list[tuple[int, dict]] = []
    for resource in resources:
        name = str(resource.get("name", ""))
        score = station_match_score(name, target.aliases)
        if score > 0:
            scored.append((score, resource))

    if not scored:
        raise RuntimeError(f"No 15-minute resource found for target: {target.canonical_name}")

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    best = [resource for score, resource in scored if score == best_score]
    if len(best) > 1:
        names = [str(item.get("name", "")) for item in best]
        raise RuntimeError(
            f"Ambiguous station match for {target.canonical_name}. Candidates: {names}"
        )
    return best[0]


def pick_column(columns: list[str], keywords: tuple[str, ...]) -> str | None:
    normalized = {col.lower(): col for col in columns}
    for key, original in normalized.items():
        if all(part in key for part in keywords):
            return original
    return None


def station_dataframe(session: requests.Session, resource: dict, target: StationTarget) -> tuple[pd.DataFrame, list[str]]:
    source_url = str(resource["url"])
    diagnostics: list[str] = []
    frame = pd.read_csv(source_url)

    timestamp_col = pick_column(list(frame.columns), ("timestamp",))
    if timestamp_col is None:
        raise RuntimeError(f"Timestamp column not found for {target.canonical_name}")

    station_col = pick_column(list(frame.columns), ("station", "name"))
    if station_col is not None and not frame.empty:
        station_values = frame[station_col].astype(str).str.lower()
        if not any(any(alias.lower() in value for alias in target.aliases) for value in station_values.head(200)):
            raise RuntimeError(
                f"Station content validation failed for {target.canonical_name} using resource {resource.get('name', '')}"
            )

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        raise RuntimeError(f"No valid timestamps for {target.canonical_name}")

    max_ts = frame["timestamp"].max()
    min_allowed = max_ts - timedelta(days=365)
    frame = frame[frame["timestamp"] >= min_allowed].copy()

    pm25_col = pick_column(list(frame.columns), ("pm2", "5"))
    pm10_col = pick_column(list(frame.columns), ("pm10",))
    no2_col = pick_column(list(frame.columns), ("no2",))
    so2_col = pick_column(list(frame.columns), ("so2",))
    co_col = pick_column(list(frame.columns), ("co", "mg"))
    ozone_col = pick_column(list(frame.columns), ("ozone",))
    at_col = pick_column(list(frame.columns), ("at",))
    rh_col = pick_column(list(frame.columns), ("rh",))

    output = pd.DataFrame(
        {
            "station_target": target.canonical_name,
            "timestamp": frame["timestamp"],
            "pm2_5": pd.to_numeric(frame[pm25_col], errors="coerce") if pm25_col else pd.NA,
            "pm10": pd.to_numeric(frame[pm10_col], errors="coerce") if pm10_col else pd.NA,
            "no2": pd.to_numeric(frame[no2_col], errors="coerce") if no2_col else pd.NA,
            "so2": pd.to_numeric(frame[so2_col], errors="coerce") if so2_col else pd.NA,
            "co_mg_m3": pd.to_numeric(frame[co_col], errors="coerce") if co_col else pd.NA,
            "ozone": pd.to_numeric(frame[ozone_col], errors="coerce") if ozone_col else pd.NA,
            "temperature_c": pd.to_numeric(frame[at_col], errors="coerce") if at_col else pd.NA,
            "humidity_pct": pd.to_numeric(frame[rh_col], errors="coerce") if rh_col else pd.NA,
            "source_url": source_url,
        }
    )

    for metric in REQUIRED_METRICS:
        if output[metric].isna().all():
            raise RuntimeError(f"Required metric {metric} is fully missing for {target.canonical_name}")

    optional_metrics = ["no2", "so2", "co_mg_m3", "ozone", "temperature_c", "humidity_pct"]
    for metric in optional_metrics:
        if output[metric].isna().all():
            diagnostics.append(f"{target.canonical_name}: optional metric '{metric}' is fully missing")

    return output, diagnostics


def quality_row(df: pd.DataFrame, station_name: str) -> dict:
    row_count = int(len(df))
    return {
        "station": station_name,
        "rows": row_count,
        "start_utc": df["timestamp"].min().isoformat() if row_count else None,
        "end_utc": df["timestamp"].max().isoformat() if row_count else None,
        "pm2_5_missing_pct": float(df["pm2_5"].isna().mean() * 100.0) if row_count else None,
        "pm10_missing_pct": float(df["pm10"].isna().mean() * 100.0) if row_count else None,
        "no2_missing_pct": float(df["no2"].isna().mean() * 100.0) if row_count else None,
        "so2_missing_pct": float(df["so2"].isna().mean() * 100.0) if row_count else None,
        "co_missing_pct": float(df["co_mg_m3"].isna().mean() * 100.0) if row_count else None,
        "ozone_missing_pct": float(df["ozone"].isna().mean() * 100.0) if row_count else None,
        "temperature_missing_pct": float(df["temperature_c"].isna().mean() * 100.0) if row_count else None,
        "humidity_missing_pct": float(df["humidity_pct"].isna().mean() * 100.0) if row_count else None,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = create_retry_session()

    metadata = load_dataset_metadata(session)
    resources = list(iter_station_resources(metadata.get("resources", [])))
    if not resources:
        raise RuntimeError("No 15-minute station resources found in CKAN dataset")

    station_frames: list[pd.DataFrame] = []
    quality_records: list[dict] = []
    selected_resources: dict[str, str] = {}
    diagnostics: list[str] = []

    for target in TARGET_STATIONS:
        matched = find_resource_for_target(resources, target)
        selected_resources[target.canonical_name] = str(matched.get("name", ""))

        df_station, station_diagnostics = station_dataframe(session, matched, target)
        station_frames.append(df_station)
        quality_records.append(quality_row(df_station, target.canonical_name))
        diagnostics.extend(station_diagnostics)

        station_file = OUTPUT_DIR / f"{slugify(target.canonical_name)}_1y.csv"
        df_station.to_csv(station_file, index=False)

    combined = pd.concat(station_frames, ignore_index=True)
    combined = combined.sort_values(["station_target", "timestamp"]).reset_index(drop=True)

    combined_csv = OUTPUT_DIR / "hyderabad_selected_stations_1y.csv"
    combined.to_csv(combined_csv, index=False)

    combined_gzip = OUTPUT_DIR / "hyderabad_selected_stations_1y.csv.gz"
    combined.to_csv(combined_gzip, index=False, compression="gzip")

    quality_df = pd.DataFrame(quality_records)
    quality_file = OUTPUT_DIR / "quality_report.csv"
    quality_df.to_csv(quality_file, index=False)

    selection_file = OUTPUT_DIR / "selected_resources.json"
    selection_file.write_text(json.dumps(selected_resources, indent=2), encoding="utf-8")

    diagnostics_file = OUTPUT_DIR / "diagnostics.log"
    diagnostics_file.write_text("\n".join(diagnostics) if diagnostics else "No diagnostics.", encoding="utf-8")

    print(f"Saved combined dataset: {combined_csv} | rows={len(combined)}")
    print(f"Saved compressed dataset: {combined_gzip}")
    print(f"Saved quality report: {quality_file}")
    print(f"Saved selected resource map: {selection_file}")
    print(f"Saved diagnostics log: {diagnostics_file}")


if __name__ == "__main__":
    main()
