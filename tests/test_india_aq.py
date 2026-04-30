from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest

import pandas as pd

from src.data.cities import ALL_MAJOR_CITIES, INDIA_MAJOR_CITIES, city_by_slug, dag_id_for_city
from src.ingestion.india_aq import (
    build_archive_path,
    incremental_window,
    normalize_observation_frame,
)


class IndiaAqIngestionTests(unittest.TestCase):
    def test_incremental_window_uses_initial_lookback_when_watermark_missing(self) -> None:
        run_end = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
        start_at, end_at = incremental_window(
            None,
            run_end=run_end,
            overlap_hours=6,
            initial_lookback_hours=24,
        )

        self.assertEqual(start_at, datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(end_at, run_end)

    def test_incremental_window_overlaps_last_observation(self) -> None:
        last_observed_at = datetime(2026, 4, 24, 18, 0, tzinfo=timezone.utc)
        run_end = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)

        start_at, end_at = incremental_window(
            last_observed_at,
            run_end=run_end,
            overlap_hours=6,
        )

        self.assertEqual(start_at, datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(end_at, run_end)

    def test_build_archive_path_uses_city_slug_and_run_date(self) -> None:
        city = INDIA_MAJOR_CITIES[3]
        archive_path = build_archive_path(
            Path("/tmp/archive"),
            city,
            "aq-20260425T120000Z",
            datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(
            archive_path.as_posix(),
            f"/tmp/archive/{city.slug}/2026/04/25/aq-20260425T120000Z.csv.gz",
        )

    def test_normalize_observation_frame_requires_fixed_schema(self) -> None:
        frame = pd.DataFrame(
            {
                "city": ["Delhi"],
                "timestamp": ["2026-04-25T00:00:00Z"],
                "pm2_5": [10.0],
                "pm10": [20.0],
                "carbon_monoxide": [0.1],
                "nitrogen_dioxide": [0.2],
                "sulphur_dioxide": [0.3],
                "ozone": [0.4],
                "us_aqi": [42],
            }
        )

        normalized = normalize_observation_frame(frame)

        self.assertEqual(list(normalized.columns), [
            "city",
            "timestamp",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi",
        ])
        self.assertEqual(str(normalized.iloc[0]["timestamp"].tzinfo), "UTC")

    def test_city_catalog_includes_global_cities_and_unique_dag_ids(self) -> None:
        catalog = ALL_MAJOR_CITIES

        self.assertGreaterEqual(len(catalog), 30)
        self.assertIn("london", city_by_slug())
        self.assertIn("new_york_city", city_by_slug())

        dag_ids = [dag_id_for_city(city) for city in catalog]
        self.assertEqual(len(dag_ids), len(set(dag_ids)))
        self.assertEqual(dag_ids[0], f"aq_{INDIA_MAJOR_CITIES[0].slug}_incremental_hourly")


if __name__ == "__main__":
    unittest.main()
