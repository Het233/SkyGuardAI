"""
================================================================================
SkyGuard AI — Missing Data & Packet Loss Injector
================================================================================
Simulates telemetry packet loss, gateway timeout, solar battery brownouts,
or cellular communication dropouts resulting in NaN observations.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseAnomalyInjector
from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent, CORE_VARIABLES


class MissingDataInjector(BaseAnomalyInjector):
    """
    Injects missing data (NaN) into sensor variables, either for a single variable or across all sensors.
    """

    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: Optional[int] = None,
        all_sensors: bool = False,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject missing values (NaN) over a specified time window.

        Args:
            df: DataFrame of AWS observations.
            target_variable: Variable to drop. If None and all_sensors=False, chosen randomly.
            station_id: Station to target. If None, chosen randomly.
            start_idx: Starting index. If None, chosen randomly.
            duration: Outage duration in steps (default: 3 to 24).
            all_sensors: If True, sets all core variables to NaN simultaneously (power outage).

        Returns:
            Tuple of (modified DataFrame copy, list of AnomalyEvent).
        """
        self._validate_input_df(df)
        df_out = df.copy()

        available_stations = df_out["station_id"].unique()
        if station_id is not None:
            if station_id not in available_stations:
                raise ValueError(f"Station '{station_id}' not found.")
            target_st = station_id
        else:
            target_st = self.rng.choice(available_stations)

        station_indices = df_out.index[df_out["station_id"] == target_st].tolist()
        n_st = len(station_indices)

        steps = duration if duration is not None else int(self.rng.randint(3, 24))
        if n_st < steps + 2:
            return df_out, []

        if start_idx is None:
            max_start = n_st - steps - 1
            rel_start = self.rng.randint(1, max(2, max_start))
        else:
            rel_start = max(0, min(start_idx, n_st - steps))

        event_indices = [station_indices[rel_start + i] for i in range(steps)]

        if all_sensors:
            target_vars = list(CORE_VARIABLES)
        else:
            target_vars = [self._select_variable(target_variable)]

        for v in target_vars:
            df_out.loc[event_indices, v] = np.nan

        # Severity based on duration and number of sensors affected
        if steps < 6 and not all_sensors:
            severity = AnomalySeverity.LOW
        elif steps < 12:
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
            anomaly_type=AnomalyType.MISSING_DATA,
            severity=severity,
            target_variables=target_vars,
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=steps,
            params={
                "all_sensors": all_sensors,
                "duration": steps,
            },
            description=f"Injected {steps}-step NaN missing data on {target_vars}."
        )

        return df_out, [event]
