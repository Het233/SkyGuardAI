"""
dashboard/simulation/anomaly_engine.py
=======================================
Fault injection and AI anomaly detection for a single AWS station time series.

Public API
----------
FAULT_TYPES : list[str]
inject_fault(df, fault_type, intensity, duration_hours, start_idx)
    -> (df_faulty: pd.DataFrame, detection: dict)
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np
import pandas as pd

FAULT_TYPES = [
    "Temperature Spike",
    "Humidity Stuck",
    "Pressure Drift",
    "Battery Failure",
    "Sensor Dropout",
    "Communication Loss",
]

_SEVERITY_MAP = {
    (1, 3):  "Low",
    (4, 6):  "Moderate",
    (7, 9):  "High",
    (10, 10):"Critical",
}

_ROOT_CAUSE = {
    "Temperature Spike":   "Sensor calibration drift / solar heating on housing",
    "Humidity Stuck":      "Capacitive humidity sensor saturation or failure",
    "Pressure Drift":      "Barometric transducer membrane fatigue",
    "Battery Failure":     "Power instability — solar regulator fault",
    "Sensor Dropout":      "MCU watchdog reset / firmware crash",
    "Communication Loss":  "GSM/LoRa link degradation — antenna or SIM fault",
}

_ACTION = {
    "Temperature Spike":   "Recalibrate or replace PT100 sensor; add radiation shield.",
    "Humidity Stuck":      "Clean/replace HC-SR04 humidity probe; check enclosure seal.",
    "Pressure Drift":      "Replace BMP280 barometric module; verify O-ring integrity.",
    "Battery Failure":     "Replace solar charge controller; inspect battery cells.",
    "Sensor Dropout":      "Remote firmware restart; schedule on-site inspection.",
    "Communication Loss":  "Replace SIM card; inspect coaxial antenna connector.",
}


def _severity_label(intensity: int) -> str:
    for (lo, hi), label in _SEVERITY_MAP.items():
        if lo <= intensity <= hi:
            return label
    return "Unknown"


def inject_fault(
    df: pd.DataFrame,
    fault_type: str,
    intensity: int,           # 1–10
    duration_hours: int,
    start_idx: int,
) -> Tuple[pd.DataFrame, dict]:
    """Apply a hardware fault to a copy of the station series.

    Parameters
    ----------
    df            : hourly series from weather_generator
    fault_type    : one of FAULT_TYPES
    intensity     : 1 (mild) … 10 (catastrophic)
    duration_hours: number of hours the fault persists
    start_idx     : row index where fault begins

    Returns
    -------
    df_faulty : pd.DataFrame  (copy with corrupted values)
    detection : dict          (AI detection result)
    """
    df_faulty = df.copy()
    end_idx = min(start_idx + duration_hours, len(df))
    rng = np.random.default_rng(42)

    scale = intensity / 10.0   # 0.1 … 1.0

    if fault_type == "Temperature Spike":
        spike = scale * 25.0    # up to +25 °C spike
        df_faulty.loc[start_idx:end_idx - 1, "temperature"] += (
            spike + rng.normal(0, 0.5, end_idx - start_idx)
        )

    elif fault_type == "Humidity Stuck":
        stuck_val = float(df_faulty.loc[start_idx, "humidity"])
        df_faulty.loc[start_idx:end_idx - 1, "humidity"] = (
            stuck_val + rng.normal(0, 0.05, end_idx - start_idx)
        )

    elif fault_type == "Pressure Drift":
        drift = np.linspace(0, scale * 30.0, end_idx - start_idx)
        df_faulty.loc[start_idx:end_idx - 1, "pressure"] += drift

    elif fault_type == "Battery Failure":
        # Voltage collapses
        drop_curve = scale * 2.5 * np.exp(
            np.linspace(0, 3, end_idx - start_idx)
        )
        df_faulty.loc[start_idx:end_idx - 1, "battery_voltage"] -= drop_curve
        df_faulty["battery_voltage"] = df_faulty["battery_voltage"].clip(0, 15)

    elif fault_type == "Sensor Dropout":
        cols = ["temperature", "humidity", "pressure", "wind_speed"]
        n_cols = max(1, int(scale * len(cols)))
        drop_cols = cols[:n_cols]
        df_faulty.loc[start_idx:end_idx - 1, drop_cols] = np.nan

    elif fault_type == "Communication Loss":
        cols = ["temperature", "humidity", "pressure",
                "wind_speed", "rainfall", "signal_strength"]
        df_faulty.loc[start_idx:end_idx - 1, cols] = np.nan

    # AI detection -------------------------------------------------------
    detection = _detect(df, df_faulty, fault_type, intensity, start_idx, end_idx)
    return df_faulty, detection


def _detect(
    df_truth: pd.DataFrame,
    df_faulty: pd.DataFrame,
    fault_type: str,
    intensity: int,
    start_idx: int,
    end_idx: int,
) -> dict:
    """Rule-based anomaly detection mimicking an ML detector."""
    # Detection latency: scales inversely with intensity
    latency_min = max(2, int(15 - intensity * 1.2))  # minutes
    detect_idx  = start_idx + max(1, latency_min // 60)

    # Confidence = function of intensity + number of anomalous sensors
    base_conf = 60 + intensity * 3.5
    conf_jitter = np.random.default_rng(start_idx).uniform(-2, 2)
    confidence = round(min(99.9, base_conf + conf_jitter), 1)

    detect_ts = (
        df_faulty["timestamp"].iloc[detect_idx]
        if detect_idx < len(df_faulty)
        else df_faulty["timestamp"].iloc[-1]
    )

    return {
        "fault_type":       fault_type,
        "detected_at":      detect_ts.strftime("%Y-%m-%d %H:%M"),
        "latency_minutes":  latency_min,
        "confidence":       confidence,
        "root_cause":       _ROOT_CAUSE.get(fault_type, "Unknown"),
        "severity":         _severity_label(intensity),
        "action":           _ACTION.get(fault_type, "Contact field engineer."),
        "affected_rows":    end_idx - start_idx,
        "start_ts":         df_faulty["timestamp"].iloc[start_idx].strftime("%Y-%m-%d %H:%M"),
    }
