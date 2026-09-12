"""
================================================================================
SkyGuard AI — Synthetic Anomaly Visualizer
================================================================================
Generates publication-quality diagnostic plots showing pristine vs. corrupted
time-series traces for each meteorological sensor anomaly class.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from anomaly_injection.taxonomy import AnomalyType


def create_visualization(
    csv_path: str = "data/synthetic/sample_synthetic_anomalies.csv",
    output_image: str = "artifacts/figures/synthetic_anomalies_overview.png",
):
    """
    Produce a multi-panel visual grid showcasing distinct synthetic anomaly injections.
    """
    if not os.path.exists(csv_path):
        print(f"[!] Error: File '{csv_path}' does not exist.")
        return

    print(f"[*] Loading dataset from {csv_path} for visualization...")
    df = pd.read_csv(csv_path)

    # Pick representative examples for distinct anomaly types
    types_to_plot = [
        AnomalyType.SPIKE.value,
        AnomalyType.DROP.value,
        AnomalyType.FROZEN_SENSOR.value,
        AnomalyType.SENSOR_DRIFT.value,
        AnomalyType.HIGH_NOISE.value,
        AnomalyType.PHYSICALLY_IMPOSSIBLE.value,
        AnomalyType.CROSS_SENSOR_INCONSISTENCY.value,
        AnomalyType.MISSING_DATA.value,
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 12), dpi=150)
    fig.suptitle("SkyGuard AI — Synthetic Anomaly Injections (Pristine vs Injected)", fontsize=16, fontweight="bold", y=0.99)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "clean": "#1f77b4",       # Calm meteorological blue
        "injected": "#d62728",    # Warning crimson
        "highlight": "#ff7f0e",   # Anomaly accent
    }

    for idx, (anom_type, ax) in enumerate(zip(types_to_plot, axes.flatten())):
        subset = df[df["anomaly_type"] == anom_type]
        if subset.empty:
            ax.text(0.5, 0.5, f"No events for {anom_type}", ha="center", va="center")
            continue

        # Select first contiguous event
        first_idx = subset.index[0]
        station = subset.loc[first_idx, "station_id"]
        st_df = df[df["station_id"] == station]

        # Context window: +/- 30 steps around the event
        window_start = max(st_df.index[0], first_idx - 25)
        window_end = min(st_df.index[-1], first_idx + 35)

        event_window = st_df.loc[window_start:window_end].copy()
        x_axis = np.arange(len(event_window))

        # Determine target variable
        target_var = subset.loc[first_idx, "anomaly_target"].split(",")[0]
        if target_var == "none" or target_var not in df.columns:
            target_var = "temperature_c"

        clean_col = f"clean_{target_var}"
        clean_vals = event_window[clean_col].values
        injected_vals = event_window[target_var].values
        is_anom = event_window["is_anomaly"].values

        # Plot clean trace
        ax.plot(x_axis, clean_vals, label="Pristine Normal", color=colors["clean"], lw=2.0, alpha=0.85, linestyle="--")

        # Plot injected trace
        ax.plot(x_axis, injected_vals, label="Injected Fault", color=colors["injected"], lw=1.8)

        # Highlight anomalous region
        anom_indices = np.where(is_anom == 1)[0]
        if len(anom_indices) > 0:
            ax.axvspan(anom_indices[0] - 0.5, anom_indices[-1] + 0.5, color=colors["highlight"], alpha=0.25, label="Ground Truth Anomaly")

        unit = "°C" if "temperature" in target_var else ("mbar" if "pressure" in target_var else "%")
        var_label = target_var.replace("_", " ").title()

        ax.set_title(f"Fault: {anom_type} ({var_label}) | Station: {station}", fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel(f"{var_label} ({unit})", fontsize=9)
        ax.set_xlabel("Relative Time Steps (Hours)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = os.path.dirname(output_image)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Saving diagnostic plot to: {output_image}")
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()
    print(f"[+] Diagnostic figure created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Synthetic Anomalies")
    parser.add_argument("--csv", type=str, default="data/synthetic/sample_synthetic_anomalies.csv")
    parser.add_argument("--out", type=str, default="artifacts/figures/synthetic_anomalies_overview.png")
    args = parser.parse_args()

    create_visualization(args.csv, args.out)
