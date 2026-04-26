# Hyderabad Station Data Analysis (Last 1 Year)

- Input rows: 105,123
- Stations: 3
- Time range: 2024-12-31T23:45:00+00:00 to 2025-12-31T23:45:00+00:00

## Storage Decision
- Use Raspberry Pi server as the system of record (InfluxDB/Timescale) for training and drift monitoring.
- Use ThingSpeak only as a dashboard mirror/quick visualization channel, not primary storage.

## Data Quality Snapshot
- ICRISAT Patancheru: rows=35041, PM2.5 missing=8.81%, PM10 missing=7.88%
- Central University (UoH): rows=35041, PM2.5 missing=20.32%, PM10 missing=25.56%
- IITH Kandi: rows=35041, PM2.5 missing=24.82%, PM10 missing=25.27%

## Generated Artifacts
- hyderabad_selected_stations_1y.csv
- hyderabad_selected_stations_1y.csv.gz
- quality_report.csv
- station_summary_stats.csv
- monthly_trends.csv
- completeness_percent.csv