"""
================================================================================
SkyGuard AI — Multivariate & Cross-Sensor Inconsistency Injectors
================================================================================
Simulates cross-sensor physical contradictions (e.g. thermal surge accompanied
by near-saturation humidity) and coordinated multi-sensor simultaneous failures.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseAnomalyInjector
from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent, CORE_VARIABLES


class CrossSensorInconsistencyInjector(BaseAnomalyInjector):
    """
    Injects thermodynamic inconsistencies where one variable changes in direct
    contradiction to meteorological coupling laws (e.g. Clausius-Clapeyron relation).
    """

    def inject(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: int = 1,
        inconsistency_mode: Optional[str] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject a cross-sensor inconsistency into a single station.

        Modes:
            - 'temp_rh_conflict': Temperature jumps +15°C while RH jumps to 98% (absurd dewpoint).
            - 'pressure_temp_decouple': Severe barometric collapse without thermal fluctuation.

        Args:
            df: DataFrame of AWS observations.
            station_id: Target station. If None, chosen randomly.
            start_idx: Target index. If None, chosen randomly.
            duration: Number of consecutive steps.
            inconsistency_mode: 'temp_rh_conflict' or 'pressure_temp_decouple'.

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
        if n_st < duration + 2:
            return df_out, []

        if start_idx is None:
            max_start = n_st - duration - 1
            rel_start = self.rng.randint(1, max(2, max_start))
        else:
            rel_start = max(0, min(start_idx, n_st - duration))

        event_indices = [station_indices[rel_start + i] for i in range(duration)]

        modes = ["temp_rh_conflict", "pressure_temp_decouple"]
        mode = inconsistency_mode if inconsistency_mode in modes else self.rng.choice(modes)

        if mode == "temp_rh_conflict":
            # Driving temperature up to high heat and RH to 99% simultaneously
            df_out.loc[event_indices, "temperature_c"] += float(self.rng.uniform(12.0, 20.0))
            df_out.loc[event_indices, "relative_humidity_pct"] = float(self.rng.uniform(96.0, 99.5))
            target_vars = ["temperature_c", "relative_humidity_pct"]
            desc = "Thermal surge combined with saturated humidity violating psychrometric balance"
        else:
            # Sudden barometric drop of -35 mbar without corresponding meteorological signs
            df_out.loc[event_indices, "air_pressure_mbar"] -= float(self.rng.uniform(30.0, 50.0))
            target_vars = ["air_pressure_mbar"]
            desc = "Isolated barometric collapse with static temperature and humidity"

        start_time = str(df_out.loc[event_indices[0], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[0])
        end_time = str(df_out.loc[event_indices[-1], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[-1])

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=AnomalyType.CROSS_SENSOR_INCONSISTENCY,
            severity=AnomalySeverity.HIGH,
            target_variables=target_vars,
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=duration,
            params={
                "mode": mode,
                "duration": duration,
            },
            description=f"Injected cross-sensor inconsistency ({desc}) across {target_vars}."
        )

        return df_out, [event]


class CoordinatedAnomalyInjector(BaseAnomalyInjector):
    """
    Injects coordinated multivariate anomalies across all three sensor variables simultaneously.
    """

    def inject(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        duration: int = 1,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject simultaneous anomalies across Temperature, Pressure, and Humidity.
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
        if n_st < duration + 2:
            return df_out, []

        if start_idx is None:
            max_start = n_st - duration - 1
            rel_start = self.rng.randint(1, max(2, max_start))
        else:
            rel_start = max(0, min(start_idx, n_st - duration))

        event_indices = [station_indices[rel_start + i] for i in range(duration)]

        # Perturb all three variables simultaneously
        t_shift = float(self.rng.uniform(10.0, 25.0) * self.rng.choice([1.0, -1.0]))
        p_shift = float(self.rng.uniform(25.0, 60.0) * self.rng.choice([1.0, -1.0]))
        rh_shift = float(self.rng.uniform(30.0, 60.0) * self.rng.choice([1.0, -1.0]))

        df_out.loc[event_indices, "temperature_c"] += t_shift
        df_out.loc[event_indices, "air_pressure_mbar"] += p_shift
        df_out.loc[event_indices, "relative_humidity_pct"] = np.clip(
            df_out.loc[event_indices, "relative_humidity_pct"] + rh_shift, 0.0, 100.0
        )

        start_time = str(df_out.loc[event_indices[0], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[0])
        end_time = str(df_out.loc[event_indices[-1], "timestamp"]) if "timestamp" in df_out.columns else str(event_indices[-1])

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=AnomalyType.COORDINATED_MULTIVARIATE,
            severity=AnomalySeverity.CRITICAL,
            target_variables=list(CORE_VARIABLES),
            start_idx=event_indices[0],
            end_idx=event_indices[-1],
            start_time=start_time,
            end_time=end_time,
            duration_steps=duration,
            params={
                "t_shift": t_shift,
                "p_shift": p_shift,
                "rh_shift": rh_shift,
                "duration": duration,
            },
            description=f"Injected coordinated multi-sensor perturbation across all core variables."
        )

        return df_out, [event]
