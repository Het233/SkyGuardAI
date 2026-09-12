"""
================================================================================
SkyGuard AI — Deterministic Quality Control (QC) Validator
================================================================================
Implements WMO/IMD-standard deterministic meteorological quality control checks:
  1. Physical plausibility boundary checks (extreme gross errors).
  2. Rate-of-change / step limits (unphysical jumps between consecutive steps).
  3. Persistence / flatline check (stuck/frozen sensors).
  4. Missingness & null checks.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from anomaly_injection.taxonomy import CORE_VARIABLES


class DeterministicQCValidator:
    """
    Tier-1 deterministic quality control rules for temperature, pressure, and humidity.
    """

    # Physical plausibility bounds (WMO & IMD surface meteorology)
    DEFAULT_BOUNDS = {
        "temperature_c": (-50.0, 60.0),
        "air_pressure_mbar": (300.0, 1100.0),
        "relative_humidity_pct": (0.0, 100.0),
    }

    # Maximum allowable rate of change between consecutive hourly observations
    DEFAULT_STEP_LIMITS = {
        "temperature_c": 12.0,       # Max 12 °C in 1 hour
        "air_pressure_mbar": 15.0,    # Max 15 mbar in 1 hour
        "relative_humidity_pct": 45.0 # Max 45 % in 1 hour
    }

    # Minimum consecutive identical readings to flag as sensor flatline/freeze
    DEFAULT_PERSISTENCE_STEPS = 6

    def __init__(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        step_limits: Optional[Dict[str, float]] = None,
        persistence_steps: int = DEFAULT_PERSISTENCE_STEPS,
    ):
        self.bounds = bounds or self.DEFAULT_BOUNDS
        self.step_limits = step_limits or self.DEFAULT_STEP_LIMITS
        self.persistence_steps = persistence_steps

    def validate(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run all deterministic quality control checks on the input DataFrame.

        Args:
            df: DataFrame containing AWS observations with 'station_id' and core variables.

        Returns:
            Tuple of:
                - is_qc_failure: Boolean pd.Series indicating whether any QC check failed (True = anomaly).
                - qc_details: pd.DataFrame containing individual test flags and failure reasons.
        """
        n_rows = len(df)
        qc_details = pd.DataFrame(index=df.index)

        # 1. Missingness check
        for var in CORE_VARIABLES:
            if var in df.columns:
                qc_details[f"{var}_is_nan"] = df[var].isna()

        # 2. Physical boundary plausibility check
        for var in CORE_VARIABLES:
            if var in df.columns and var in self.bounds:
                low, high = self.bounds[var]
                qc_details[f"{var}_out_of_bounds"] = (df[var] < low) | (df[var] > high)

        # 3. Rate-of-change and persistence checks (must be calculated per station)
        rate_flags = pd.Series(False, index=df.index)
        persistence_flags = pd.Series(False, index=df.index)

        if "station_id" in df.columns:
            stations = df["station_id"].unique()
            for st in stations:
                st_mask = df["station_id"] == st
                st_indices = df.index[st_mask]

                for var in CORE_VARIABLES:
                    if var not in df.columns:
                        continue

                    series = df.loc[st_indices, var]

                    # Step check: |x_t - x_{t-1}| > limit
                    step_diff = series.diff().abs()
                    limit = self.step_limits.get(var, np.inf)
                    rate_fail = step_diff > limit
                    qc_details.loc[st_indices, f"{var}_rate_of_change_exceeded"] = rate_fail
                    rate_flags.loc[st_indices] = rate_flags.loc[st_indices] | rate_fail

                    # Persistence check: consecutive identical values
                    # Compare each value with previous; True if identical
                    is_same = (series == series.shift(1))
                    # Rolling sum of identical transitions over window
                    consec_count = is_same.rolling(window=self.persistence_steps - 1, min_periods=self.persistence_steps - 1).sum()
                    frozen = consec_count >= (self.persistence_steps - 1)
                    
                    # Also flag the preceding points that formed the freeze
                    if frozen.any():
                        frozen_expanded = frozen.copy()
                        for shift_back in range(1, self.persistence_steps):
                            frozen_expanded = frozen_expanded | frozen.shift(-shift_back).fillna(False)
                        qc_details.loc[st_indices, f"{var}_frozen"] = frozen_expanded
                        persistence_flags.loc[st_indices] = persistence_flags.loc[st_indices] | frozen_expanded
                    else:
                        qc_details.loc[st_indices, f"{var}_frozen"] = False

        # Aggregate overall QC failure
        any_failure = pd.Series(False, index=df.index)
        for col in qc_details.columns:
            any_failure = any_failure | qc_details[col].fillna(False)

        qc_details["qc_passed"] = ~any_failure
        qc_details["qc_anomaly"] = any_failure.astype(int)

        return any_failure, qc_details
