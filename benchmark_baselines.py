"""
================================================================================
SkyGuard AI — Baseline Benchmarking Harness
================================================================================
Evaluates and benchmarks all Tier-1 and Tier-2 baseline detectors:
  1. Deterministic Quality Control (WMO/IMD physical bounds, step limits, flatlines)
  2. Rolling Z-Score Detector
  3. Rolling IQR / Tukey's Fences Detector
  4. EWMA Residual Detector
  5. Composite Baseline Detector (Ensemble)

Outputs rigorous comparative metrics (Precision, Recall, F1, PR-AUC, Latency, FPR).
================================================================================
"""

import os
import sys
import argparse
import time
import json
import pandas as pd
import numpy as np

from baseline import (
    DeterministicQCValidator,
    ZScoreDetector,
    IQRDetector,
    EWMADetector,
    CompositeBaselineDetector,
)
from evaluation import compute_binary_metrics, compute_type_breakdown


def run_benchmarks(csv_path: str, output_json: str, output_md: str):
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 78)
    print("  [+] SKYGUARD AI -- BASELINE DETECTOR BENCHMARK")
    print("=" * 78)

    if not os.path.exists(csv_path):
        print(f"[!] Error: Evaluation dataset '{csv_path}' not found.")
        sys.exit(1)

    print(f"[*] Loading evaluation dataset: {csv_path}")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df):,} observations in {time.time() - t0:.2f}s.")

    if "is_anomaly" not in df.columns:
        print("[!] Error: Dataset lacks ground-truth 'is_anomaly' column.")
        sys.exit(1)

    y_true = df["is_anomaly"].values
    total_anomalies = int(np.sum(y_true))
    anomaly_rate = total_anomalies / len(y_true) * 100.0
    num_stations = df["station_id"].nunique() if "station_id" in df.columns else 1

    print(f"[*] Evaluation ground truth: {total_anomalies:,} anomalies ({anomaly_rate:.2f}%) across {num_stations} station(s).")

    # Instantiate detectors
    detectors = {
        "Deterministic QC": DeterministicQCValidator(),
        "Rolling Z-Score": ZScoreDetector(window=24, threshold=3.5),
        "Rolling IQR": IQRDetector(window=48, multiplier=1.8),
        "EWMA Residuals": EWMADetector(span=24, sigma_threshold=3.5),
        "Composite Baseline": CompositeBaselineDetector(z_threshold=3.5, iqr_multiplier=1.8),
    }

    results = {}
    type_breakdowns = {}

    print("\n[*] Benchmarking detectors...")
    for name, detector in detectors.items():
        print(f"  -> Running {name}...", end="", flush=True)
        t_start = time.perf_counter()

        if name == "Deterministic QC":
            flags, details = detector.validate(df)
            scores = flags.astype(float)
        elif name == "Composite Baseline":
            flags, scores, components = detector.detect(df)
        else:
            flags, scores = detector.detect(df)

        latency_sec = time.perf_counter() - t_start
        latency_us_per_sample = (latency_sec / len(df)) * 1_000_000

        metrics = compute_binary_metrics(
            y_true=y_true,
            y_pred=flags,
            y_score=scores,
            num_stations=num_stations,
        )
        metrics["latency_us_per_sample"] = round(latency_us_per_sample, 2)
        metrics["total_time_seconds"] = round(latency_sec, 2)

        results[name] = metrics

        # Compute type breakdown
        df_temp = df.copy()
        df_temp["pred_flag"] = flags
        type_breakdowns[name] = compute_type_breakdown(df_temp, y_pred_col="pred_flag")

        print(f" Done in {latency_sec:.2f}s (F1: {metrics['f1_score']:.4f}, Recall: {metrics['recall']:.4f})")

    # Print markdown table
    print("\n" + "=" * 78)
    print("  [*] BASELINE DETECTORS COMPARISON TABLE")
    print("=" * 78)

    headers = [
        "Method",
        "Precision",
        "Recall",
        "F1-Score",
        "PR-AUC",
        "FPR (%)",
        "False Alerts/Day",
        "Latency (µs)",
    ]

    row_format = "{:<20} | {:>9} | {:>7} | {:>8} | {:>7} | {:>7} | {:>16} | {:>12}"
    print(row_format.format(*headers))
    print("-" * 105)

    md_table_rows = []
    for name, m in results.items():
        pr_auc_str = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "—"
        fpr_pct_str = f"{m['fpr'] * 100:.2f}%"
        row_str = row_format.format(
            name,
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            pr_auc_str,
            fpr_pct_str,
            f"{m['false_alerts_per_station_day']:.2f}",
            f"{m['latency_us_per_sample']:.1f}",
        )
        print(row_str)

        md_table_rows.append(
            f"| **{name}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {pr_auc_str} | {fpr_pct_str} | {m['false_alerts_per_station_day']:.2f} | {m['latency_us_per_sample']:.1f} µs |"
        )

    print("=" * 78)

    # Print breakdown by anomaly type for Composite Baseline
    print("\n  [*] COMPOSITE BASELINE RECALL BY FAULT TYPE:")
    print("-" * 55)
    for a_type, data in sorted(type_breakdowns["Composite Baseline"].items(), key=lambda x: x[1]["recall"], reverse=True):
        print(f"    - {a_type:<28}: {data['recall'] * 100:>6.1f}% ({data['detected']}/{data['total_instances']})")
    print("-" * 55)

    # Save JSON results
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "breakdown": type_breakdowns}, f, indent=2)
    print(f"\n[+] Detailed benchmark JSON saved to: {output_json}")

    # Save Markdown report
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    md_content = f"""# SkyGuard AI — Baseline Detectors Benchmark Report

**Dataset:** `{csv_path}`  
**Observations:** {len(df):,} rows | **Stations:** {num_stations} | **Total Anomalies:** {total_anomalies:,} ({anomaly_rate:.2f}%)

## Performance Comparison

| Method | Precision | Recall | F1-Score | PR-AUC | False Positive Rate | False Alerts/Stn-Day | Latency/Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
""" + "\n".join(md_table_rows) + """

## Composite Baseline Recall by Anomaly Type

| Anomaly Type | Injected Instances | Detected Instances | Detection Recall |
| :--- | :---: | :---: | :---: |
""" + "\n".join([
        f"| `{t}` | {d['total_instances']} | {d['detected']} | **{d['recall'] * 100:.1f}%** |"
        for t, d in sorted(type_breakdowns["Composite Baseline"].items(), key=lambda x: x[1]["recall"], reverse=True)
    ]) + """

## Key Findings & Research Insights
1. **Deterministic QC** excels at gross errors (Physical Bounds, NaNs, Frozen Sensors, Spikes) with near-zero false alarms.
2. **Statistical Detectors (Z-score & IQR)** catch dynamic departures but have difficulty distinguishing gradual drift from seasonal cycles.
3. **Composite Baseline** achieves a strong baseline foundation (**F1 benchmark**) for the Machine Learning models (Isolation Forest, Autoencoder, and Residual Predictors) to improve upon in Phase 4.
"""
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Benchmark Markdown report saved to: {output_md}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Baseline Detectors")
    parser.add_argument("--input", type=str, default="data/synthetic/sample_synthetic_anomalies.csv")
    parser.add_argument("--out-json", type=str, default="artifacts/benchmark_results.json")
    parser.add_argument("--out-md", type=str, default="artifacts/baseline_benchmark.md")
    args = parser.parse_args()

    run_benchmarks(args.input, args.out_json, args.out_md)
