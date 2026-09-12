"""
================================================================================
SkyGuard AI — Frozen Sensor Anomaly Injector
================================================================================
Simulates frozen/stuck sensor readings (flatlining) caused by ADC conversion
freezes, I2C/SPI bus hangs, hardware latch-ups, or firmware lockups.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseAnomalyInjector
from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent


class FrozenSensorInjector(BaseAnomalyInjector):
    """
    Injects flatline/stuck values where sensor readings remain unchanged over multiple hours.
    """

    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: Optional[int] = None,
        stuck_value: Optional[float] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject a frozen sensor reading into a time series.

        Args:
            df: DataFrame of AWS observations.
            target_variable: One of 'temperature_c', 'air_pressure_mbar', 'relative_humidity_pct'.
            station_id: Station to target. If None, chosen randomly.
            start_idx: Starting index. If None, chosen randomly.
            duration: Number of consecutive frozen steps (default: 6 to 36 steps).
            stuck_value: Constant value to hold. If None, uses the value at start_idx.

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

        steps = duration if duration is not None else int(self.rng.randint(6, 36))
        if n_st < steps + 2:
            return df_out, []

        if start_idx is None:
            max_start = n_st - steps - 1
            rel_start = self.rng.randint(1, max(2, max_start))
        else:
            rel_start = max(0, min(start_idx, n_st - steps))

        event_indices = [station_indices[rel_start + i] for i in range(steps)]

        # Determine stuck value
        if stuck_value is not None:
            val = float(stuck_value)
        else:
            val = float(df_out.loc[event_indices[0], var])

        df_out.loc[event_indices, var] = val

        # Severity based on duration of flatline
        if steps < 12:
            severity = AnomalySeverity.MEDIUM
        elif steps < 24:
            severity = AnomalySeverity.HIGH
        else:
            severity = AnomalySeverity.CRITICAL

        start_time = str(df_out.loc[event_indices[0], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[0])
        end_time = str(df_out.loc[event_indices[-1], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[-1])

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=AnomalyType.FROZEN_SENSOR,
            severity=severity,
            target_variables=[var],
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=steps,
            params={
                "stuck_value": val,
                "duration": steps,
            },
            description=f"Injected {steps}-step flatline at value {val:.2f} on {var}."
        )

        return df_out, [event]
