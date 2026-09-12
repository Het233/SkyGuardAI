"""
================================================================================
SkyGuard AI — Spike and Drop Anomaly Injector
================================================================================
Simulates transient high-magnitude spikes and dips caused by electrical surges,
ADC bit glitches, inductive motor switching, or lightning EMI.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseAnomalyInjector
from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent


class SpikeInjector(BaseAnomalyInjector):
    """
    Injects transient single-point or short-burst spikes and drops into sensor time series.
    """

    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: int = 1,
        direction: Optional[str] = None,  # "positive", "negative", or None (random)
        sigma_multiplier: Optional[float] = None,
        magnitude: Optional[float] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject a spike or drop into the specified station and variable.

        Args:
            df: DataFrame of AWS observations.
            target_variable: One of 'temperature_c', 'air_pressure_mbar', 'relative_humidity_pct'.
            station_id: Station to target. If None, chosen randomly.
            start_idx: Row index within the station subset. If None, chosen randomly.
            duration: Number of consecutive steps (typically 1 to 3).
            direction: "positive" for SPIKE, "negative" for DROP, or None for random.
            sigma_multiplier: Multiplier of variable standard deviation (default: 4.0 to 7.0).
            magnitude: Fixed absolute offset to add/subtract. Overrides sigma_multiplier if set.

        Returns:
            Tuple of (modified DataFrame copy, list of AnomalyEvent).
        """
        self._validate_input_df(df)
        df_out = df.copy()
        var = self._select_variable(target_variable)

        # Select station
        available_stations = df_out["station_id"].unique()
        if station_id is not None:
            if station_id not in available_stations:
                raise ValueError(f"Station '{station_id}' not found in DataFrame.")
            target_st = station_id
        else:
            target_st = self.rng.choice(available_stations)

        station_indices = df_out.index[df_out["station_id"] == target_st].tolist()
        n_st = len(station_indices)
        if n_st < duration + 2:
            return df_out, []

        # Select start position
        if start_idx is None:
            max_start = n_st - duration - 1
            rel_start = self.rng.randint(1, max(2, max_start))
        else:
            rel_start = max(0, min(start_idx, n_st - duration))

        event_indices = [station_indices[rel_start + i] for i in range(duration)]

        # Calculate standard deviation of the variable for this station
        st_series = df_out.loc[station_indices, var].dropna()
        var_std = float(st_series.std()) if len(st_series) > 1 and st_series.std() > 0 else 1.0

        # Determine direction and magnitude
        if direction is None:
            is_positive = bool(self.rng.choice([True, False]))
        else:
            is_positive = (direction.lower() == "positive")

        if magnitude is not None:
            val_shift = float(magnitude) if is_positive else -float(magnitude)
            ratio = abs(val_shift) / var_std
        else:
            sigma = float(sigma_multiplier) if sigma_multiplier is not None else float(self.rng.uniform(3.5, 7.5))
            val_shift = sigma * var_std if is_positive else -sigma * var_std
            ratio = sigma

        anomaly_type = AnomalyType.SPIKE if is_positive else AnomalyType.DROP
        severity = self._determine_severity(ratio)

        # Apply perturbation
        original_vals = df_out.loc[event_indices, var].values
        perturbed_vals = original_vals + val_shift

        # If humidity, keep physically bound within standard checks unless it's an impossible fault
        if var == "relative_humidity_pct":
            perturbed_vals = np.clip(perturbed_vals, 0.0, 100.0)

        df_out.loc[event_indices, var] = perturbed_vals

        start_time = str(df_out.loc[event_indices[0], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[0])
        end_time = str(df_out.loc[event_indices[-1], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[-1])

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=anomaly_type,
            severity=severity,
            target_variables=[var],
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=duration,
            params={
                "magnitude": float(val_shift),
                "sigma_multiplier": float(ratio),
                "direction": "positive" if is_positive else "negative",
            },
            description=f"Injected {anomaly_type.value} of shift {val_shift:+.2f} on {var} across {duration} steps."
        )

        return df_out, [event]
