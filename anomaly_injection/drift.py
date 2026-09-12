"""
================================================================================
SkyGuard AI — Sensor Drift & Constant Offset Injectors
================================================================================
Simulates gradual calibration drift (dust/fouling, aging sensing material, battery
decay) and systematic constant offsets (miscalibration, uncalibrated replacement).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseAnomalyInjector
from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent


class DriftInjector(BaseAnomalyInjector):
    """
    Injects gradual calibration drift (linear or non-linear) across a continuous window.
    """

    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: Optional[int] = None,
        drift_rate: Optional[float] = None,  # per step rate
        max_drift: Optional[float] = None,
        drift_type: str = "linear",  # "linear" or "exponential"
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject progressive sensor drift over a window of time.

        Args:
            df: DataFrame of AWS observations.
            target_variable: One of 'temperature_c', 'air_pressure_mbar', 'relative_humidity_pct'.
            station_id: Station to target. If None, chosen randomly.
            start_idx: Starting index. If None, chosen randomly.
            duration: Number of steps over which drift occurs (e.g. 24 to 168 hours).
            drift_rate: Amount added per step. If None, computed from max_drift or variable std.
            max_drift: Maximum cumulative drift at the end of the window.
            drift_type: "linear" or "exponential".

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

        steps = duration if duration is not None else int(self.rng.randint(24, 120))
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

        # Determine direction (+ or -)
        direction_sign = float(self.rng.choice([1.0, -1.0]))

        if max_drift is not None:
            total_drift = abs(max_drift) * direction_sign
        elif drift_rate is not None:
            total_drift = drift_rate * steps * direction_sign
        else:
            total_drift = (var_std * float(self.rng.uniform(2.5, 5.0))) * direction_sign

        # Compute step offsets
        step_arr = np.arange(steps)
        if drift_type == "exponential":
            # Growth towards asymptote
            drift_profile = total_drift * (1.0 - np.exp(-3.0 * step_arr / max(1, steps)))
        else:
            # Linear growth
            drift_profile = total_drift * (step_arr / max(1, steps))

        original_vals = df_out.loc[event_indices, var].values
        perturbed_vals = original_vals + drift_profile

        if var == "relative_humidity_pct":
            perturbed_vals = np.clip(perturbed_vals, 0.0, 100.0)

        df_out.loc[event_indices, var] = perturbed_vals

        ratio = abs(total_drift) / var_std
        severity = self._determine_severity(ratio)

        start_time = str(df_out.loc[event_indices[0], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[0])
        end_time = str(df_out.loc[event_indices[-1], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[-1])

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=AnomalyType.SENSOR_DRIFT,
            severity=severity,
            target_variables=[var],
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=steps,
            params={
                "total_drift": float(total_drift),
                "drift_type": drift_type,
                "duration": steps,
            },
            description=f"Injected {drift_type} drift of {total_drift:+.2f} on {var} over {steps} steps."
        )

        return df_out, [event]


class ConstantOffsetInjector(BaseAnomalyInjector):
    """
    Injects a sustained constant calibration offset (step bias) over a prolonged duration.
    """

    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: Optional[int] = None,
        offset: Optional[float] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject a sustained constant offset into a sensor reading.
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

        steps = duration if duration is not None else int(self.rng.randint(12, 72))
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

        if offset is not None:
            bias = float(offset)
        else:
            sign = float(self.rng.choice([1.0, -1.0]))
            bias = sign * var_std * float(self.rng.uniform(2.0, 4.5))

        original_vals = df_out.loc[event_indices, var].values
        perturbed_vals = original_vals + bias

        if var == "relative_humidity_pct":
            perturbed_vals = np.clip(perturbed_vals, 0.0, 100.0)

        df_out.loc[event_indices, var] = perturbed_vals

        ratio = abs(bias) / var_std
        severity = self._determine_severity(ratio)

        start_time = str(df_out.loc[event_indices[0], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[0])
        end_time = str(df_out.loc[event_indices[-1], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[-1])

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=AnomalyType.CONSTANT_OFFSET,
            severity=severity,
            target_variables=[var],
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=steps,
            params={
                "offset": float(bias),
                "duration": steps,
            },
            description=f"Injected constant offset of {bias:+.2f} on {var} over {steps} steps."
        )

        return df_out, [event]
