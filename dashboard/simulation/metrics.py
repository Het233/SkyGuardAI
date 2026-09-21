"""
dashboard/simulation/metrics.py
=================================
Fleet-level statistics derived from the 826-station EDA dataset.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def fleet_summary(df_stations: pd.DataFrame) -> dict:
    """Compute fleet KPIs from the station-level dataframe.

    Parameters
    ----------
    df_stations : pd.DataFrame
        Output of load_aws_dataset() — one row per station.

    Returns
    -------
    dict with keys:
        total, healthy, degrading, critical, avg_health,
        detection_accuracy (static showcase value),
        active_faults
    """
    total   = len(df_stations)
    healthy = int((df_stations["status"].isin(["EXCELLENT", "GOOD"])).sum())
    degrading = int((df_stations["status"] == "DEGRADING").sum())
    critical  = int((df_stations["status"] == "CRITICAL").sum())
    avg_health = round(float(df_stations["health_score"].mean()), 1)
    active_faults = critical + int(degrading * 0.4)

    return {
        "total":              total,
        "healthy":            healthy,
        "degrading":          degrading,
        "critical":           critical,
        "avg_health":         avg_health,
        "detection_accuracy": 96.4,     # AI model accuracy (demo value)
        "active_faults":      active_faults,
    }


def station_health_score(series_df: pd.DataFrame) -> float:
    """Estimate health of a station from its telemetry series.

    Uses data completeness and sensor value plausibility as proxy.
    Returns 0–100 float.
    """
    if series_df.empty:
        return 0.0
    sensor_cols = ["temperature", "humidity", "pressure",
                   "battery_voltage", "signal_strength"]
    completeness = series_df[sensor_cols].notna().mean().mean()
    # Check battery: penalise if mean voltage < 11.5 V
    bat_penalty = 0.0
    if "battery_voltage" in series_df.columns:
        mean_bat = series_df["battery_voltage"].mean()
        if mean_bat < 11.5:
            bat_penalty = (11.5 - mean_bat) * 10
    score = max(0.0, completeness * 100 - bat_penalty)
    return round(float(score), 1)
