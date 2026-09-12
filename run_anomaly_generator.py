"""
================================================================================
SkyGuard AI — Synthetic Anomaly Generator CLI
================================================================================
Command-line utility to generate ground-truth labeled synthetic anomaly datasets
for testing and training the SkyGuard AI anomaly detection models.

Usage Examples:
  # 1. Standard run on sample dataset (5% anomalies, seed 42):
  python run_anomaly_generator.py --input sample_aws_weather_data.csv --output data/synthetic/sample_synthetic_anomalies.csv

  # 2. Custom anomaly rate and specific stations:
  python run_anomaly_generator.py --input sample_aws_weather_data.csv --rate 0.08 --stations IMD_AWS_0009,IMD_AWS_0008

  # 3. Save detailed JSON ground-truth registry:
  python run_anomaly_generator.py --input sample_aws_weather_data.csv --output data/synthetic/sample_synthetic.csv --save-registry data/synthetic/event_registry.json
================================================================================
"""

import os
import sys
import argparse
import time
import pandas as pd

from anomaly_injection import SyntheticAnomalyGenerator, AnomalyType, AnomalySeverity


def parse_args():
    parser = argparse.ArgumentParser(
        description="SkyGuard AI — Synthetic Anomaly Generator for Meteorological AWS Data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="sample_aws_weather_data.csv",
        help="Path to clean input CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/synthetic_aws_weather_data.csv",
        help="Path where output augmented CSV should be saved.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Fraction of observations to convert into anomalies (default: 0.05 = 5%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible injection (default: 42).",
    )
    parser.add_argument(
        "--stations",
        type=str,
        default=None,
        help="Optional comma-separated list of station IDs to process.",
    )
    parser.add_argument(
        "--save-registry",
        type=str,
        default=None,
        help="Optional JSON file path to save detailed event-by-event metadata.",
    )
    return parser.parse_args()


def print_banner():
    print("=" * 78)
    print("  [+] SKYGUARD AI -- SYNTHETIC ANOMALY GENERATOR")
    print("=" * 78)


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    print_banner()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input dataset '{args.input}' not found.")
        sys.exit(1)

    print(f"[*] Loading input dataset from: {args.input}")
    t0 = time.time()
    df = pd.read_csv(args.input)
    load_time = time.time() - t0
    print(f"[+] Loaded {len(df):,} rows across {df['station_id'].nunique()} stations in {load_time:.2f}s.")

    station_subset = None
    if args.stations:
        station_subset = [s.strip() for s in args.stations.split(",") if s.strip()]
        print(f"[*] Filtering to {len(station_subset)} station(s): {station_subset}")

    # Prepare generator
    print(f"[*] Initializing SyntheticAnomalyGenerator (rate={args.rate*100:.1f}%, seed={args.seed})...")
    generator = SyntheticAnomalyGenerator(anomaly_rate=args.rate, seed=args.seed)

    print("[*] Injecting synthetic anomalies across stations...")
    t1 = time.time()
    df_augmented, events = generator.generate(df, station_subset=station_subset)
    gen_time = time.time() - t1

    report = generator.get_summary_report(df_augmented)

    print("\n" + "=" * 78)
    print("  [*] GENERATION SUMMARY REPORT")
    print("=" * 78)
    print(f"  Total Observations:        {report['total_observations']:,}")
    print(f"  Total Anomalous Rows:      {report['total_anomalous_rows']:,}")
    print(f"  Actual Anomaly Rate:       {report['actual_anomaly_rate_pct']:.2f}%")
    print(f"  Total Anomaly Events:      {report['total_events_injected']:,}")
    print(f"  Execution Time:            {gen_time:.2f}s")
    print("-" * 78)
    print("  Breakdown by Anomaly Type:")
    for anom_type, count in sorted(report["distribution_by_type"].items(), key=lambda x: x[1], reverse=True):
        print(f"    - {anom_type:<28} : {count:>6,} rows")
    print("-" * 78)
    print("  Breakdown by Severity:")
    for sev, count in sorted(report["distribution_by_severity"].items(), key=lambda x: x[1], reverse=True):
        print(f"    - {sev:<28} : {count:>6,} rows")
    print("=" * 78)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\n[*] Saving augmented dataset to: {args.output}")
    df_augmented.to_csv(args.output, index=False)
    out_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"[+] Saved successfully ({out_size_mb:.2f} MB).")

    # Optionally save event registry
    registry_path = args.save_registry
    if registry_path is None:
        # Default registry path alongside output
        base_name = os.path.splitext(args.output)[0]
        registry_path = f"{base_name}_events.json"

    reg_dir = os.path.dirname(registry_path)
    if reg_dir:
        os.makedirs(reg_dir, exist_ok=True)

    print(f"[*] Saving ground-truth event registry to: {registry_path}")
    generator.save_event_registry(registry_path)
    print("[+] Event registry saved successfully.")
    print("[+] Anomaly Generation Phase Complete.\n")


if __name__ == "__main__":
    main()
