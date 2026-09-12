"""
================================================================================
SkyGuard AI — All-India AWS Weather Data Collector
================================================================================
Author: SkyGuard AI Team
Data Source: Open-Meteo Historical Weather Archive API
Coverage: 826 Automatic Weather Stations across all 36 Indian States & UTs (100% India)
Target Variables:
  - Temperature (°C)             [Open-Meteo: temperature_2m]
  - Surface Air Pressure (mbar)  [Open-Meteo: surface_pressure]
  - Relative Humidity (%)        [Open-Meteo: relative_humidity_2m]

Key Capabilities:
  1. Auto-Resume / Checkpointing: If internet disconnects or script is stopped,
     it seamlessly resumes from the exact station it left off without duplicates.
  2. Memory Efficient Streaming: Writes station data incrementally to CSV.
     Memory usage stays below 50 MB even for 25+ million rows.
  3. Built-in Polite Rate Limiting & Exponential Backoff for HTTP 429.
  4. Real-time Progress Bar & Estimated Time of Arrival (ETA).

Usage Examples:
  # 1. Download all 826 stations from 2023 to present:
  python collect_aws_dataset.py

  # 2. Quick test with first 5 stations:
  python collect_aws_dataset.py --limit 5 --output test_sample.csv

  # 3. Custom date range (e.g., 2023 to 2025):
  python collect_aws_dataset.py --start-date 2023-01-01 --end-date 2025-12-31
================================================================================
"""

import os
import sys
import time
import argparse
import datetime
import requests
import pandas as pd

# Default configurations
DEFAULT_STATIONS_FILE = "complete_all_india_aws_stations.csv"
DEFAULT_OUTPUT_FILE = "aws_weather_data_all_india_2023_present.csv"
CHECKPOINT_FILE = "download_checkpoint.txt"
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = datetime.date.today().strftime("%Y-%m-%d")

API_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_station_metadata(filepath):
    """Load the master 826-station registry."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Station registry not found: '{filepath}'. "
            "Please ensure complete_all_india_aws_stations.csv is in the workspace directory."
        )

    print(f"[*] Loading station registry from: {filepath}")
    df = pd.read_csv(filepath)
    required_cols = ["station_id", "station_name", "state", "district", "latitude", "longitude", "elevation_m"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in station file: {missing_cols}")

    stations = df.drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    print(f"[+] Successfully loaded {len(stations)} stations across {stations['state'].nunique()} States & UTs.")
    return stations


def get_completed_station_ids(checkpoint_file):
    """Retrieve list of station IDs that have already been written to disk."""
    if not os.path.exists(checkpoint_file):
        return set()
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        completed = {line.strip() for line in f if line.strip()}
    return completed


def record_station_completed(checkpoint_file, station_id):
    """Mark a station ID as completed in the checkpoint tracking file."""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(f"{station_id}\n")


def fetch_weather_for_station(station, start_date, end_date, max_retries=5):
    """
    Fetch hourly historical weather observations for a single AWS station.
    Retries automatically with exponential backoff on rate-limits (HTTP 429) or network drops.
    """
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure",
        "timezone": "Asia/Kolkata"
    }

    url = (
        f"{API_ARCHIVE_URL}?"
        f"latitude={params['latitude']}&longitude={params['longitude']}&"
        f"start_date={params['start_date']}&end_date={params['end_date']}&"
        f"hourly={params['hourly']}&timezone={params['timezone']}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                hourly = data.get("hourly", {})
                timestamps = hourly.get("time", [])
                if not timestamps:
                    return None

                df = pd.DataFrame({
                    "station_id": station["station_id"],
                    "station_name": station["station_name"],
                    "state": station["state"],
                    "district": station["district"],
                    "latitude": station["latitude"],
                    "longitude": station["longitude"],
                    "elevation_m": station["elevation_m"],
                    "timestamp": timestamps,
                    "temperature_c": hourly.get("temperature_2m"),
                    "air_pressure_mbar": hourly.get("surface_pressure"),
                    "relative_humidity_pct": hourly.get("relative_humidity_2m")
                })
                return df

            elif response.status_code == 429:
                sleep_seconds = 60 * attempt
                print(f" [!] Open-Meteo Rate Limit (HTTP 429). Pacing gently: waiting {sleep_seconds}s...")
                time.sleep(sleep_seconds)
            else:
                print(f" [!] HTTP {response.status_code} (attempt {attempt}): {response.text[:80]}")
                time.sleep(5 * attempt)

        except requests.exceptions.RequestException as err:
            print(f" [!] Network issue on attempt {attempt}: {err}")
            time.sleep(5 * attempt)

    print(f" [X] Failed to fetch station {station['station_id']} after {max_retries} retries.")
    return None


def run_pipeline(stations_path, output_path, start_date, end_date, limit=None, delay=1.5):
    """Execute the end-to-end extraction, transformation, and incremental CSV writing."""
    stations = load_station_metadata(stations_path)
    if limit:
        stations = stations.head(limit)
        print(f"[*] Limiting execution to first {limit} stations.")

    completed_ids = get_completed_station_ids(CHECKPOINT_FILE)
    pending_stations = [s for _, s in stations.iterrows() if s["station_id"] not in completed_ids]

    total_stations = len(stations)
    total_pending = len(pending_stations)

    print("-" * 75)
    print(f" Target Output File : {output_path}")
    print(f" Date Range         : {start_date} to {end_date}")
    print(f" Total Stations     : {total_stations}")
    print(f" Already Cached     : {len(completed_ids)}")
    print(f" Stations to Fetch  : {total_pending}")
    print("-" * 75)

    if total_pending == 0:
        print("[+] All stations are already downloaded in the dataset!")
        return

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    start_time = time.time()
    success_count = 0

    for idx, stn in enumerate(pending_stations, start=1):
        stn_start_time = time.time()
        stn_id = stn["station_id"]
        stn_name = stn["station_name"]
        state = stn["state"]

        # Calculate progress and ETA
        elapsed = time.time() - start_time
        avg_time = (elapsed / success_count) if success_count > 0 else 0
        remaining_sec = avg_time * (total_pending - idx + 1)
        eta_display = str(datetime.timedelta(seconds=int(remaining_sec))) if success_count > 0 else "calculating..."
        pct = (idx / total_pending) * 100

        print(f"[{idx:>3}/{total_pending}] ({pct:5.1f}%) Fetching {stn_name} ({state}) [ETA: {eta_display}]... ", end="", flush=True)

        df_stn = fetch_weather_for_station(stn, start_date, end_date)

        if df_stn is not None and not df_stn.empty:
            # Stream directly to CSV on disk
            df_stn.to_csv(
                output_path,
                mode="a",
                header=not file_exists,
                index=False,
                encoding="utf-8"
            )
            file_exists = True
            record_station_completed(CHECKPOINT_FILE, stn_id)
            success_count += 1
            duration = time.time() - stn_start_time
            print(f"OK ({len(df_stn):,} rows in {duration:.1f}s)")
        else:
            print("FAILED")

        # Polite delay to remain well within API guidelines
        time.sleep(delay)

    total_duration = time.time() - start_time
    print("=" * 75)
    print(f"[+] Download complete! Successfully retrieved {success_count} stations.")
    print(f"[+] Total execution time: {total_duration / 60:.2f} minutes.")
    print(f"[+] Final dataset stored at: {output_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download multi-year hourly AWS weather data across India from Open-Meteo."
    )
    parser.add_argument(
        "--stations", default=DEFAULT_STATIONS_FILE,
        help="Path to CSV containing station metadata (default: complete_all_india_aws_stations.csv)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_FILE,
        help="Path to target CSV dataset (default: aws_weather_data_all_india_2023_present.csv)"
    )
    parser.add_argument(
        "--start-date", default=DEFAULT_START_DATE,
        help="Start date in YYYY-MM-DD format (default: 2023-01-01)"
    )
    parser.add_argument(
        "--end-date", default=DEFAULT_END_DATE,
        help="End date in YYYY-MM-DD format (default: today's date)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of stations to fetch (useful for testing)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Delay in seconds between API calls (default: 1.5s)"
    )

    args = parser.parse_args()
    run_pipeline(
        stations_path=args.stations,
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        delay=args.delay
    )
