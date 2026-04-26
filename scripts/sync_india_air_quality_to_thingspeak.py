from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.cities import ALL_MAJOR_CITIES, INDIA_MAJOR_CITIES, city_by_slug
from src.data.live_air_quality import HOURLY_FIELDS, fetch_latest_city_observation
from src.integrations.thingspeak import ThingSpeakClient

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_FIELD_NAMES = [
    "field1",
    "field2",
    "field3",
    "field4",
    "field5",
    "field6",
    "field7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync live AQ samples to ThingSpeak city dashboards.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument("--cities", default="", help="Comma-separated city names or slugs")
    parser.add_argument("--publish", action="store_true", help="Enable live ThingSpeak publishing")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads without publishing")
    parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Request retry count")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_city_subset(raw_value: str) -> list:
    if not raw_value.strip():
        return list(INDIA_MAJOR_CITIES)

    registry = city_by_slug()
    requested = []
    for token in raw_value.split(","):
        normalized = token.strip().lower().replace(" ", "_")
        if normalized in registry:
            requested.append(registry[normalized])
            continue
        match = next((city for city in ALL_MAJOR_CITIES if city.name.lower() == token.strip().lower()), None)
        if match is None:
            raise ValueError(f"Unknown city: {token.strip()}")
        requested.append(match)
    return requested


def build_field_payload(
    observation: dict[str, object],
    metric_names: list[str],
    field_names: list[str],
    city_name: str,
) -> dict[str, object]:
    if len(metric_names) < len(field_names):
        raise ValueError("publish_fields must provide at least as many metrics as ThingSpeak fields")

    payload = {}
    for index, field_name in enumerate(field_names):
        metric_name = metric_names[index]
        value = observation.get(metric_name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError(f"{city_name}: missing value for {metric_name}")
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"{city_name}: non-finite value for {metric_name}")
        payload[field_name] = value
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    thingspeak_config = config.get("thingspeak", {}) if isinstance(config, dict) else {}
    enabled = bool(thingspeak_config.get("enabled", False))
    effective_dry_run = args.dry_run or not args.publish or not enabled or bool(thingspeak_config.get("dry_run", False))
    channel_strategy = str(thingspeak_config.get("channel_strategy", "one_channel_per_city"))
    if channel_strategy != "one_channel_per_city":
        raise SystemExit(f"Unsupported ThingSpeak channel_strategy: {channel_strategy}")

    field_names = list(thingspeak_config.get("field_names", DEFAULT_FIELD_NAMES))
    if len(field_names) < len(HOURLY_FIELDS):
        raise SystemExit("ThingSpeak field_names must provide at least 7 fields")
    publish_metrics = list(thingspeak_config.get("publish_fields", list(HOURLY_FIELDS)))

    timeout = int(thingspeak_config.get("timeout_seconds", args.timeout))
    retries = int(thingspeak_config.get("retry_count", args.retries))
    selected_cities = resolve_city_subset(args.cities or str(",".join(thingspeak_config.get("publish_cities", []))))
    client = ThingSpeakClient(timeout=timeout, retries=retries)

    if args.publish and not enabled:
        raise SystemExit("ThingSpeak publishing is disabled in config.yaml. Set thingspeak.enabled=true to publish live.")

    failures: list[str] = []
    try:
        for city in selected_cities:
            env_prefix = str(thingspeak_config.get("write_key_env_prefix", "THINGSPEAK_WRITE_KEY_"))
            env_name = f"{env_prefix}{city.slug.upper()}"
            write_key = os.getenv(env_name, "").strip()
            if not write_key and not effective_dry_run:
                failures.append(f"{city.name}: missing {env_name}")
                continue

            observation = fetch_latest_city_observation(city, timeout=timeout)
            payload = build_field_payload(observation, publish_metrics, field_names, city.name)
            status = f"city={city.name};timestamp={pd.Timestamp(observation['timestamp']).isoformat()}"

            if effective_dry_run:
                print(f"DRY RUN {city.name}: {payload} | status={status}")
                continue

            result = client.publish(write_key=write_key, fields=payload, status=status)
            if result.success:
                print(f"Published {city.name}: entry_id={result.entry_id}")
            else:
                failures.append(f"{city.name}: {result.message}")
                print(f"Failed {city.name}: {result.message}")
    finally:
        client.close()

    if failures:
        failure_text = "\n".join(failures)
        raise SystemExit(f"ThingSpeak sync completed with failures:\n{failure_text}")


if __name__ == "__main__":
    main()