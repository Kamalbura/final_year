# ThingSpeak Live Setup

This project now supports live publishing of city-level air-quality readings to ThingSpeak.

## Recommended Layout

Use one ThingSpeak channel per city.

Reason:
- Each channel only supports 8 fields.
- The live payload uses 7 metrics per city.
- A separate channel per city keeps the dashboard simple and avoids field exhaustion.

## Field Mapping

The sync script publishes these values in order:
- field1 = pm2_5
- field2 = pm10
- field3 = carbon_monoxide
- field4 = nitrogen_dioxide
- field5 = sulphur_dioxide
- field6 = ozone
- field7 = us_aqi

The status text includes the city name and timestamp.

## Enable Live Publishing

1. Edit [config.yaml](../../config.yaml) and set:
   - `thingspeak.enabled: true`
   - keep `thingspeak.dry_run: false`

2. Set one write key environment variable per city channel:
   - `THINGSPEAK_WRITE_KEY_DELHI`
   - `THINGSPEAK_WRITE_KEY_MUMBAI`
   - `THINGSPEAK_WRITE_KEY_BENGALURU`
   - `THINGSPEAK_WRITE_KEY_HYDERABAD`
   - `THINGSPEAK_WRITE_KEY_CHENNAI`
   - `THINGSPEAK_WRITE_KEY_KOLKATA`
   - `THINGSPEAK_WRITE_KEY_PUNE`
   - `THINGSPEAK_WRITE_KEY_AHMEDABAD`
   - `THINGSPEAK_WRITE_KEY_JAIPUR`
   - `THINGSPEAK_WRITE_KEY_LUCKNOW`
   - `THINGSPEAK_WRITE_KEY_SURAT`
   - `THINGSPEAK_WRITE_KEY_KANPUR`
   - `THINGSPEAK_WRITE_KEY_NAGPUR`
   - `THINGSPEAK_WRITE_KEY_BHOPAL`
   - `THINGSPEAK_WRITE_KEY_VISAKHAPATNAM`

3. Run the sync command:

```powershell
C:/Users/burak/miniconda3/Scripts/conda.exe run -p C:/Users/burak/miniconda3/envs/dl-env python scripts/sync_india_air_quality_to_thingspeak.py --publish
```

## Dry-Run First

Use dry-run mode before the first live publish:

```powershell
C:/Users/burak/miniconda3/Scripts/conda.exe run -p C:/Users/burak/miniconda3/envs/dl-env python scripts/sync_india_air_quality_to_thingspeak.py --cities Hyderabad --dry-run
```

## Dashboard Notes

- Build one dashboard per city or one multi-dashboard page that links to each city channel.
- Keep update cadence at hourly or slower unless your upstream data frequency changes.
- If you later publish forecasts, use a second channel so live observations and predictions stay separate.
