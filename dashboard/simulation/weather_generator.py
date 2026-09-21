"""
dashboard/simulation/weather_generator.py
==========================================
Deterministic 3-year hourly telemetry generator for a single IMD AWS station.
Uses the station's real latitude, longitude, and elevation from the EDA dataset
to produce physically plausible seasonal/diurnal patterns.

Public API
----------
generate_station_series(station_row, start, end) -> pd.DataFrame
    station_row : pd.Series  (one row from load_aws_dataset())
    start / end : str  ISO-8601 date strings  e.g. "2023-01-01"
    Returns a DataFrame with columns:
        timestamp, station_id, temperature, humidity, pressure,
        wind_speed, rainfall, battery_voltage, solar_voltage, signal_strength
"""

from __future__ import annotations

import hashlib
from typing import Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_LAPSE_RATE   = 6.5          # °C per 1000 m
_STD_PRESSURE = 1013.25      # hPa at sea level
_GRAVITY_SCALE = 5.256       # dimensionless exponent in barometric formula


def _station_seed(station_id: str) -> int:
    """Deterministic seed from station ID so every run gives identical data."""
    h = hashlib.md5(station_id.encode()).hexdigest()
    return int(h, 16) % (2 ** 31)


def _baseline_temperature(lat: float, elev: float) -> float:
    """Annual-mean temperature at this location."""
    # India ranges ~6°N–37°N.  Rough empirical linear fit across the subcontinent.
    T_sea_level = 32.0 - 0.45 * (lat - 6.0)   # hotter in south, cooler north
    T_elev_corrected = T_sea_level - _LAPSE_RATE * elev / 1000.0
    return float(T_elev_corrected)


def _barometric_pressure(elev: float) -> float:
    """Mean pressure at given elevation (m)."""
    return _STD_PRESSURE * ((1 - 0.0000226 * elev) ** _GRAVITY_SCALE)


def generate_station_series(
    station_row: "pd.Series",
    start: str = "2023-01-01",
    end: str   = "2025-12-31 23:00",
) -> pd.DataFrame:
    """Generate deterministic 3-year hourly telemetry for one AWS station.

    Parameters
    ----------
    station_row : pd.Series
        One row from load_aws_dataset().  Must have station_id, latitude,
        longitude, elevation, temperature, humidity, pressure.
    start, end : str
        ISO-8601 date strings delimiting the simulation window (inclusive).

    Returns
    -------
    pd.DataFrame
        Hourly rows with columns: timestamp, station_id, temperature, humidity,
        pressure, wind_speed, rainfall, battery_voltage, solar_voltage,
        signal_strength.
    """
    sid   = str(station_row["station_id"])
    lat   = float(station_row.get("latitude",  20.0))
    lon   = float(station_row.get("longitude", 78.0))
    elev  = float(station_row.get("elevation",  0.0))

    # Real sensor baseline from dataset (used to anchor the series)
    real_temp = float(station_row.get("temperature", _baseline_temperature(lat, elev)))
    real_hum  = float(station_row.get("humidity",  65.0))
    real_pres = float(station_row.get("pressure",  _barometric_pressure(elev)))

    rng = np.random.default_rng(_station_seed(sid))

    # ---- Time axis ---------------------------------------------------------
    timestamps = pd.date_range(start=start, end=end, freq="h")
    n = len(timestamps)
    # Convert to plain numpy arrays so .clip() / arithmetic works everywhere
    t_year  = (timestamps.dayofyear.to_numpy() - 1) / 365.25   # 0..1 annual
    t_day   = timestamps.hour.to_numpy()       / 24.0           # 0..1 diurnal
    t_lin   = np.arange(n) / n                                  # 0..1 aging

    # ---- Temperature -------------------------------------------------------
    T_mean = real_temp
    # Seasonal: amplitude ~10°C peak-to-peak (summer=max at day ~200 / ~Jul)
    seasonal_amp = max(5.0, 10.0 - 0.15 * abs(lat - 20))
    T_seasonal   = seasonal_amp * np.sin(2 * np.pi * (t_year - 0.45))
    # Diurnal: ±4°C swing; peaks at 14:00 LST
    T_diurnal    = 4.0 * np.sin(2 * np.pi * (t_day - 0.17))
    # Noise
    T_noise      = rng.normal(0, 0.6, n)
    temperature  = T_mean + T_seasonal + T_diurnal + T_noise

    # ---- Humidity ----------------------------------------------------------
    H_base = real_hum
    # Monsoon boost: Jun-Sep, centered at day ~213 (Aug 1)
    monsoon = np.where(
        (timestamps.month >= 6) & (timestamps.month <= 9),
        np.clip(18.0 * np.sin(np.pi * ((timestamps.dayofyear.to_numpy() - 152) / 122.0)), 0, 1),
        0.0,
    )
    # Inverse of temp anomaly: warmer → drier (approx -1.5% per °C)
    T_anomaly  = temperature - T_mean
    H_temp_inv = -1.5 * T_anomaly
    # Diurnal dew-point effect: higher humidity at dawn
    H_diurnal  = -8.0 * np.sin(2 * np.pi * (t_day - 0.17))
    H_noise    = rng.normal(0, 2.0, n)
    humidity   = (H_base + monsoon + H_temp_inv + H_diurnal + H_noise).clip(5, 100)

    # ---- Pressure ----------------------------------------------------------
    P_base    = real_pres
    # Diurnal semi-diurnal tide: ±1 hPa, twice daily
    P_diurnal = 1.2 * np.sin(4 * np.pi * t_day) + 0.4 * np.sin(2 * np.pi * t_day)
    # Slow synoptic variation: smooth random walk (AR process)
    ar_noise  = rng.normal(0, 0.15, n)
    P_ar      = np.zeros(n)
    for i in range(1, n):
        P_ar[i] = 0.98 * P_ar[i - 1] + ar_noise[i]
    pressure  = P_base + P_diurnal + P_ar

    # ---- Wind speed --------------------------------------------------------
    W_base  = 3.5 + 2.0 * np.sin(2 * np.pi * t_year + 0.8)   # seasonal
    W_gust  = rng.exponential(0.8, n)
    wind_speed = (W_base + W_gust).clip(0.2, 35.0)

    # ---- Rainfall ----------------------------------------------------------
    # Probability of rain per hour: highest Jun-Sep, depends on region
    # Western/Eastern coast and NE India get more rain
    coast_boost = 0.04 if (lon < 76 or lon > 86 or lat > 26) else 0.0
    p_monsoon = np.where(
        (timestamps.month >= 6) & (timestamps.month <= 9), 0.10 + coast_boost, 0.01
    )
    rain_occurs  = rng.random(n) < p_monsoon
    rain_amount  = rng.gamma(shape=1.5, scale=2.5, size=n) * rain_occurs
    rainfall     = rain_amount.round(1)

    # ---- Battery voltage ---------------------------------------------------
    # 12V lead-acid battery: starts at 12.8 V, degrades ~0.3 V over 3 years
    V_start = 12.80
    V_end   = 12.50
    V_trend = V_start + (V_end - V_start) * t_lin
    # Solar charging: raises voltage during daylight by up to 0.4 V
    V_solar_boost = 0.4 * np.clip(np.sin(np.pi * t_day), 0, 1)
    # Short-term fluctuation
    V_noise = rng.normal(0, 0.04, n)
    battery_voltage = (V_trend + V_solar_boost + V_noise).round(2)

    # ---- Solar panel voltage -----------------------------------------------
    # 18V panel nominal; diurnal parabola × cloud factor (humid → more cloud)
    cloud_factor = 1 - (humidity / 200)                       # 0.5..1.0
    solar_base   = 18.0 * np.clip(np.sin(np.pi * t_day), 0, 1)
    solar_noise  = rng.normal(0, 0.3, n)
    solar_voltage = (solar_base * cloud_factor + solar_noise).clip(0, 20.5).round(2)

    # ---- Signal strength (RSSI dBm) ----------------------------------------
    # Baseline depends on elevation (higher = clearer LoS) and lat scatter
    sig_base = -68.0 - 0.005 * elev + rng.normal(0, 1.0)
    sig_ar   = np.zeros(n)
    s_noise  = rng.normal(0, 1.5, n)
    for i in range(1, n):
        sig_ar[i] = 0.92 * sig_ar[i - 1] + s_noise[i]
    signal_strength = (sig_base + sig_ar).clip(-95, -45).round(1)

    return pd.DataFrame({
        "timestamp":       timestamps,
        "station_id":      sid,
        "temperature":     temperature.round(2),
        "humidity":        humidity.round(1),
        "pressure":        pressure.round(2),
        "wind_speed":      wind_speed.round(2),
        "rainfall":        rainfall,
        "battery_voltage": battery_voltage,
        "solar_voltage":   solar_voltage,
        "signal_strength": signal_strength,
    })
