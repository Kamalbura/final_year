"""
Monitor Kaggle benchmark kernel status and orchestrate forecast ingestion.

This script:
1. Checks the status of the Kaggle GPU benchmark kernel
2. Downloads outputs when the kernel succeeds
3. Loads forecasts into the database
4. Validates dashboard API access
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Kaggle kernel and orchestrate forecast ingestion.")
    parser.add_argument("--kernel", default="kamalbura/aqi-gpu-benchmark-3-city-model-zoo", help="Kaggle kernel slug")
    parser.add_argument("--output-dir", default="outputs/kaggle_benchmarks", help="Output directory for downloaded files")
    parser.add_argument("--poll-interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--max-wait", type=int, default=3600, help="Max wait time in seconds")
    parser.add_argument("--dashboard-url", default="http://localhost:3000", help="Dashboard base URL for validation")
    parser.add_argument("--skip-poll", action="store_true", help="Skip polling, assume kernel succeeded")
    return parser.parse_args()


def check_kernel_status(kernel_slug: str) -> dict[str, object]:
    """Check Kaggle kernel status via CLI."""
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle CLI failed: {result.stderr}")
        
        # Parse the output, which is typically JSON
        status_text = result.stdout.strip()
        try:
            return json.loads(status_text)
        except json.JSONDecodeError:
            # Fallback: parse basic status from text
            return {"status": status_text, "raw": result.stdout}
    except subprocess.TimeoutExpired:
        raise RuntimeError("Kaggle CLI timeout")
    except FileNotFoundError:
        raise RuntimeError("kaggle CLI not found; install via: pip install kaggle")


def poll_kernel_completion(kernel_slug: str, poll_interval: int, max_wait: int) -> bool:
    """Poll kernel status until it completes or timeout."""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            status = check_kernel_status(kernel_slug)
            print(f"[{time.time() - start_time:.0f}s] Kernel status: {status}")
            
            # Check for completion
            status_str = str(status).lower()
            if "complete" in status_str or "success" in status_str:
                print("✓ Kernel completed successfully")
                return True
            elif "error" in status_str or "failed" in status_str:
                print("✗ Kernel failed")
                return False
            
            time.sleep(poll_interval)
        except RuntimeError as e:
            print(f"Error checking status: {e}")
            time.sleep(poll_interval)
    
    print(f"✗ Kernel did not complete within {max_wait}s")
    return False


def download_kernel_outputs(kernel_slug: str, output_dir: Path) -> list[Path]:
    """Download outputs from Kaggle kernel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "output", kernel_slug, "-p", str(output_dir)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")
        
        print(f"✓ Downloaded outputs to {output_dir}")
        
        # List downloaded files
        downloaded = list(output_dir.glob("**/*"))
        for path in sorted(downloaded):
            if path.is_file():
                print(f"  - {path.relative_to(output_dir)}")
        
        return [p for p in downloaded if p.is_file()]
    except subprocess.TimeoutExpired:
        raise RuntimeError("Download timeout")
    except FileNotFoundError:
        raise RuntimeError("kaggle CLI not found")


def load_forecasts(forecast_csv: Path, dsn: str | None = None) -> dict[str, object]:
    """Load forecasts into database."""
    script = REPO_ROOT / "scripts" / "load_forecasts_to_db.py"
    cmd = [sys.executable, str(script), "--csv", str(forecast_csv)]
    if dsn:
        cmd.extend(["--dsn", dsn])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Load failed: {result.stderr}")
    
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"stdout": result.stdout}


def validate_dashboard_api(dashboard_url: str, cities: list[str] = None) -> dict[str, object]:
    """Validate that dashboard API can access forecasts."""
    if cities is None:
        cities = ["delhi", "hyderabad", "bengaluru"]
    
    results = {}
    for city in cities:
        slug = city.lower().replace(" ", "-")
        try:
            url = f"{dashboard_url}/api/predictions/{slug}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            has_forecasts = bool(data.get("timeline"))
            model_name = data.get("model", {}).get("name")
            results[city] = {
                "status": "ok",
                "has_forecasts": has_forecasts,
                "model": model_name,
                "forecast_count": len([t for t in data.get("timeline", []) if t.get("predicted_aqi")]),
            }
        except Exception as e:
            results[city] = {"status": "error", "error": str(e)}
    
    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("Kaggle Benchmark Monitor & Forecast Ingestion")
    print("=" * 70)
    
    # Step 1: Check kernel status
    print(f"\n[1/4] Checking kernel status: {args.kernel}")
    if not args.skip_poll:
        if not poll_kernel_completion(args.kernel, args.poll_interval, args.max_wait):
            print("✗ Kernel did not complete; exiting")
            sys.exit(1)
    else:
        print("⊘ Skipping poll (--skip-poll)")
    
    # Step 2: Download outputs
    print(f"\n[2/4] Downloading kernel outputs")
    try:
        files = download_kernel_outputs(args.kernel, output_dir)
        if not files:
            print("✗ No output files downloaded")
            sys.exit(1)
    except RuntimeError as e:
        print(f"✗ Download failed: {e}")
        sys.exit(1)
    
    # Step 3: Load forecasts
    print(f"\n[3/4] Loading forecasts into database")
    forecast_csv = output_dir / "forecast_rows.csv"
    if not forecast_csv.exists():
        print(f"✗ forecast_rows.csv not found at {forecast_csv}")
        sys.exit(1)
    
    try:
        dsn = os.environ.get("DATABASE_URL")
        result = load_forecasts(forecast_csv, dsn)
        print(f"✓ Loaded: {json.dumps(result, indent=2)}")
    except RuntimeError as e:
        print(f"✗ Load failed: {e}")
        sys.exit(1)
    
    # Step 4: Validate dashboard
    print(f"\n[4/4] Validating dashboard API at {args.dashboard_url}")
    try:
        validation = validate_dashboard_api(args.dashboard_url)
        print(f"✓ Validation results:")
        for city, result in validation.items():
            if result["status"] == "ok":
                print(f"  {city}: {result['forecast_count']} forecasts via {result['model']}")
            else:
                print(f"  {city}: {result['error']}")
    except Exception as e:
        print(f"⚠ Validation error: {e}")
    
    print("\n" + "=" * 70)
    print("✓ Orchestration complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
