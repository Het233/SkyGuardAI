"""
================================================================================
SkyGuard AI — Physically Impossible Values Injector
================================================================================
Simulates complete transducer breakdown, short-circuits, or open-circuits
generating readings outside physically plausible Earth meteorological boundaries.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseAnomalyInjector
from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent


class PhysicalBoundaryInjector(BaseAnomalyInjector):
    """
    Injects readings that violate basic thermodynamic and meteorological physical limits.
    """

    # Earth meteorological plausibility boundaries
    PHYSICAL_LIMITS = {
        "temperature_c": (-50.0, 60.0),        # Valid Earth surface range
        "air_pressure_mbar": (300.0, 1100.0),   # Valid tropospheric station range
        "relative_humidity_pct": (0.0, 100.0),   # Physical definition of RH
    }

    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: int = 1,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject physically impossible values into a sensor record.

        Args:
            df: DataFrame of AWS observations.
            target_variable: One of 'temperature_c', 'air_pressure_mbar', 'relative_humidity_pct'.
            station_id: Station to target. If None, chosen randomly.
            start_idx: Target index. If None, chosen randomly.
            duration: Number of consecutive steps (typically 1 to 3).

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
        if n_st < duration + 2:
            return df_out, []

        if start_idx is None:
            max_start = n_st - duration - 1
            rel_start = self.rng.randint(1, max(2, max_start))
        else:
            rel_start = max(0, min(start_idx, n_st - duration))

        event_indices = [station_indices[rel_start + i] for i in range(duration)]

        # Generate impossible value based on variable
        if var == "relative_humidity_pct":
            # Either negative or well above 100%
            if self.rng.choice([True, False]):
                impossible_val = float(self.rng.uniform(105.0, 150.0))
            else:
                impossible_val = float(self.rng.uniform(-40.0, -1.0))
        elif var == "air_pressure_mbar":
            if self.rng.choice([True, False]):
                # Negative pressure or near vacuum
                impossible_val = float(self.rng.uniform(-200.0, 150.0))
            else:
                # Extreme hyperbaric pressure
                impossible_val = float(self.rng.uniform(1200.0, 2000.0))
        else:  # temperature_c
            if self.rng.choice([True, False]):
                # Superheated reading (ADC shorted to VCC)
                impossible_val = float(self.rng.uniform(75.0, 150.0))
            else:
                # Impossible freezing (ADC open-circuit or pull-down)
                impossible_val = float(self.rng.uniform(-100.0, -65.0))

        df_out.loc[event_indices, var] = impossible_val

        start_time = str(df_out.loc[event_indices[0], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[0])
        end_time = str(df_out.loc[event_indices[-1], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[-1])

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=AnomalyType.PHYSICALLY_IMPOSSIBLE,
            severity=AnomalySeverity.CRITICAL,
            target_variables=[var],
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=duration,
            params={
                "injected_value": impossible_val,
                "duration": duration,
                "physical_limits": self.PHYSICAL_LIMITS[var],
            },
            description=f"Injected physically impossible value {impossible_val:.2f} on {var} violating {self.PHYSICAL_LIMITS[var]}."
        )

        return df_out, [event]
