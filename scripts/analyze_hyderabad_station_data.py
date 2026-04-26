from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/hyderabad_station_aq_1y/hyderabad_selected_stations_1y.csv")
OUTPUT_DIR = Path("data/hyderabad_station_aq_1y")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required_input_columns = {
        "station_target",
        "timestamp",
        "pm2_5",
        "pm10",
        "no2",
        "so2",
        "co_mg_m3",
        "ozone",
        "temperature_c",
        "humidity_pct",
    }
    missing_input = sorted(required_input_columns - set(df.columns))
    if missing_input:
        raise RuntimeError(f"Input dataset schema mismatch. Missing columns: {missing_input}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    numeric_cols = ["pm2_5", "pm10", "no2", "so2", "co_mg_m3", "ozone", "temperature_c", "humidity_pct"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    station_summary = (
        df.groupby("station_target", dropna=False)[numeric_cols]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    station_summary.columns = ["_".join([c for c in tup if c]).strip("_") for tup in station_summary.columns]
    station_summary.to_csv(OUTPUT_DIR / "station_summary_stats.csv", index=False)

    df["month"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)
    monthly_trends = (
        df.groupby(["station_target", "month"], dropna=False)[["pm2_5", "pm10", "no2", "so2"]]
        .mean()
        .reset_index()
        .sort_values(["station_target", "month"])
    )
    monthly_trends.to_csv(OUTPUT_DIR / "monthly_trends.csv", index=False)

    completeness = (
        df.groupby("station_target", dropna=False)[numeric_cols]
        .apply(lambda g: 100.0 - (g.isna().mean() * 100.0))
        .reset_index()
    )
    completeness.to_csv(OUTPUT_DIR / "completeness_percent.csv", index=False)

    notes = []
    notes.append("# Hyderabad Station Data Analysis (Last 1 Year)")
    notes.append("")
    notes.append(f"- Input rows: {len(df):,}")
    notes.append(f"- Stations: {df['station_target'].nunique()}")
    notes.append(f"- Time range: {df['timestamp'].min().isoformat()} to {df['timestamp'].max().isoformat()}")
    notes.append("")
    notes.append("## Storage Decision")
    notes.append("- Use Raspberry Pi server as the system of record (InfluxDB/Timescale) for training and drift monitoring.")
    notes.append("- Use ThingSpeak only as a dashboard mirror/quick visualization channel, not primary storage.")
    notes.append("")
    notes.append("## Data Quality Snapshot")

    q = pd.read_csv(OUTPUT_DIR / "quality_report.csv")
    required_quality_columns = {"station", "rows", "pm2_5_missing_pct", "pm10_missing_pct"}
    missing_quality = sorted(required_quality_columns - set(q.columns))
    if missing_quality:
        raise RuntimeError(f"Quality report schema mismatch. Missing columns: {missing_quality}")
    for _, row in q.iterrows():
        notes.append(
            f"- {row['station']}: rows={int(row['rows'])}, PM2.5 missing={row['pm2_5_missing_pct']:.2f}%, PM10 missing={row['pm10_missing_pct']:.2f}%"
        )

    notes.append("")
    notes.append("## Generated Artifacts")
    notes.append("- hyderabad_selected_stations_1y.csv")
    notes.append("- hyderabad_selected_stations_1y.csv.gz")
    notes.append("- quality_report.csv")
    notes.append("- station_summary_stats.csv")
    notes.append("- monthly_trends.csv")
    notes.append("- completeness_percent.csv")

    (OUTPUT_DIR / "analysis_notes.md").write_text("\n".join(notes), encoding="utf-8")

    print("Saved station_summary_stats.csv")
    print("Saved monthly_trends.csv")
    print("Saved completeness_percent.csv")
    print("Saved analysis_notes.md")


if __name__ == "__main__":
    main()
