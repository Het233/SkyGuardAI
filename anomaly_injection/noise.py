"""
================================================================================
SkyGuard AI — High Noise & Jitter Anomaly Injector
================================================================================
Simulates high-variance electrical noise and measurement jitter caused by
degraded analog shielding, loose connection resistance, or electromagnetic pickup.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseAnomalyInjector
from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent


class NoiseInjector(BaseAnomalyInjector):
    """
    Injects high-variance noise (Gaussian or uniform) into a specified time window.
    """

    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: Optional[int] = None,
        noise_type: str = "gaussian",  # "gaussian" or "uniform"
        noise_std_multiplier: Optional[float] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject high-frequency noise into the sensor signal.

        Args:
            df: DataFrame of AWS observations.
            target_variable: One of 'temperature_c', 'air_pressure_mbar', 'relative_humidity_pct'.
            station_id: Station to target. If None, chosen randomly.
            start_idx: Starting index. If None, chosen randomly.
            duration: Window length in steps (default: 12 to 48).
            noise_type: 'gaussian' or 'uniform'.
            noise_std_multiplier: Multiplier of variable standard deviation for noise sigma.

        Returns:
            Tuple of (modified DataFrame copy, list of AnomalyEvent).
        """
        self._validate_input_df(df)
        df_out = df.copy()
        var = self._select_variable(target_variable)

        available_stations = df_out["station_id"].unique()
        if station_id is not None:
            if station_id not in available_stations:
                raise ValueError(f"Station '{station_id}' not found.")
            target_st = station_id
        else:
            target_st = self.rng.choice(available_stations)

        station_indices = df_out.index[df_out["station_id"] == target_st].tolist()
        n_st = len(station_indices)

        steps = duration if duration is not None else int(self.rng.randint(12, 48))
        if n_st < steps + 2:
            return df_out, []

        if start_idx is None:
            max_start = n_st - steps - 1
            rel_start = self.rng.randint(1, max(2, max_start))
        else:
            rel_start = max(0, min(start_idx, n_st - steps))

        event_indices = [station_indices[rel_start + i] for i in range(steps)]

        st_series = df_out.loc[station_indices, var].dropna()
        var_std = float(st_series.std()) if len(st_series) > 1 and st_series.std() > 0 else 1.0

        multiplier = float(noise_std_multiplier) if noise_std_multiplier is not None else float(self.rng.uniform(1.8, 3.5))
        sigma = multiplier * var_std

        if noise_type == "uniform":
            noise_vals = self.rng.uniform(-np.sqrt(3) * sigma, np.sqrt(3) * sigma, size=steps)
        else:
            noise_vals = self.rng.normal(0, sigma, size=steps)

        original_vals = df_out.loc[event_indices, var].values
        perturbed_vals = original_vals + noise_vals

        if var == "relative_humidity_pct":
            perturbed_vals = np.clip(perturbed_vals, 0.0, 100.0)

        df_out.loc[event_indices, var] = perturbed_vals

        severity = self._determine_severity(multiplier)

        start_time = str(df_out.loc[event_indices[0], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[0])
        end_time = str(df_out.loc[event_indices[-1], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[-1])

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=AnomalyType.HIGH_NOISE,
            severity=severity,
            target_variables=[var],
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=steps,
            params={
                "noise_type": noise_type,
                "noise_std_multiplier": float(multiplier),
                "noise_sigma": float(sigma),
                "duration": steps,
            },
            description=f"Injected {noise_type} noise with sigma={sigma:.2f} on {var} over {steps} steps."
        )

        return df_out, [event]
