from __future__ import annotations

import argparse
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi continuous runtime loop for 7-day AQI forecasting.")
    parser.add_argument("--train-cmd", required=True, help="Command to train models")
    parser.add_argument("--forecast-cmd", required=True, help="Command to generate 7-day forecast")
    parser.add_argument("--monitor-cmd", required=True, help="Command to run drift monitor/retrain")
    parser.add_argument("--sync-cmd", default="", help="Optional ThingSpeak sync command")
    parser.add_argument("--interval-seconds", type=int, default=3600, help="Loop interval in seconds")
    parser.add_argument("--retrain-every-hours", type=int, default=24, help="Scheduled full retrain interval")
    return parser.parse_args()


def run_step(command: str, name: str) -> None:
    if not command.strip():
        return
    print(f"Running {name}: {command}")
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        print(f"Step failed: {name} (exit={result.returncode})")


def main() -> None:
    args = parse_args()
    cycle = 0

    while True:
        cycle += 1
        print(f"Starting cycle {cycle}")

        if cycle == 1 or (cycle * args.interval_seconds) % (args.retrain_every_hours * 3600) == 0:
            run_step(args.train_cmd, "scheduled_train")

        run_step(args.forecast_cmd, "forecast")
        run_step(args.monitor_cmd, "drift_monitor")
        run_step(args.sync_cmd, "thingspeak_sync")

        print(f"Cycle {cycle} completed. Sleeping {args.interval_seconds} seconds.")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
