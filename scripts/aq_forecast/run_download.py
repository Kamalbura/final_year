"""
run_download.py — Execute data collection for all Hyderabad stations.

Downloads 2 years of hourly air quality + meteorological data from
Open-Meteo APIs and saves per-station CSVs to data/raw/.
"""
import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.aq_forecast.data.download import download_all_station_data

if __name__ == "__main__":
    print("=" * 60)
    print("  AQI Data Collection — Hyderabad, Telangana")
    print("  Source: Open-Meteo Air Quality + Archive APIs")
    print("  Period: 2023-01-01 to 2024-12-31 (2 years)")
    print("  Stations: 7 CPCB CAAQMS stations")
    print("=" * 60)

    combined_df = download_all_station_data(
        start_date="2023-01-01",
        end_date="2024-12-31",
    )

    print(f"\nDownload complete!")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Columns: {combined_df.columns.tolist()}")
    print(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"  Stations: {combined_df['station'].nunique()}")

    # Quick quality check
    print(f"\n  Missing values per column:")
    null_counts = combined_df.isnull().sum()
    for col, cnt in null_counts.items():
        if cnt > 0:
            pct = 100.0 * cnt / len(combined_df)
            print(f"    {col}: {cnt} ({pct:.1f}%)")

    print(f"\n  AQI distribution:")
    if "AQI_Category" in combined_df.columns:
        print(combined_df["AQI_Category"].value_counts().to_string())
