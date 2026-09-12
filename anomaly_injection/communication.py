"""
================================================================================
SkyGuard AI — Communication Corruption Anomaly Injector
================================================================================
Simulates transmission errors, packet duplication, bit flips, sign inversions,
and scale errors arising in cellular/satellite AWS communication links.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseAnomalyInjector
from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent


class CommunicationCorruptionInjector(BaseAnomalyInjector):
    """
    Injects packet-level telemetry corruption (sign flips, scale/decimal shift, bit inversion).
    """

    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        station_id: Optional[str] = None,
        start_idx: Optional[int] = None,
        corruption_mode: Optional[str] = None,  # "sign_flip", "scale_shift", "duplicate_jitter"
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject communication corruptions into sensor packets.

        Args:
            df: DataFrame of AWS observations.
            target_variable: Variable to corrupt. If None, chosen randomly.
            station_id: Target station. If None, chosen randomly.
            start_idx: Target index. If None, chosen randomly.
            corruption_mode: 'sign_flip' (e.g. +25 -> -25), 'scale_shift' (e.g. * 10 or / 10),
                             or None for random selection.

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
        if n_st < 5:
            return df_out, []

        if start_idx is None:
            rel_idx = self.rng.randint(1, n_st - 2)
        else:
            rel_idx = max(0, min(start_idx, n_st - 1))

        idx = station_indices[rel_idx]
        current_val = float(df_out.loc[idx, var])

        modes = ["sign_flip", "scale_shift", "decimal_error"]
        mode = corruption_mode if corruption_mode in modes else self.rng.choice(modes)

        if mode == "sign_flip":
            corrupted_val = -current_val
            desc = f"Sign flip from {current_val} to {corrupted_val}"
        elif mode == "scale_shift":
            scale_factor = float(self.rng.choice([10.0, 0.1, 100.0]))
            corrupted_val = current_val * scale_factor
            desc = f"Scale error (factor {scale_factor}) from {current_val} to {corrupted_val}"
        else:  # decimal_error / bit flip
            corrupted_val = current_val + 256.0 * float(self.rng.choice([1.0, -1.0]))
            desc = f"Byte bit-flip from {current_val} to {corrupted_val}"

        df_out.loc[idx, var] = corrupted_val

        start_time = str(df_out.loc[idx, "timestamp"]) if "timestamp" in df_out.columns else str(idx)

        event = AnomalyEvent(
            event_id=self._generate_event_id(),
            station_id=target_st,
            anomaly_type=AnomalyType.COMMUNICATION_CORRUPTION,
            severity=AnomalySeverity.CRITICAL,
            target_variables=[var],
            start_idx=idx,
            end_idx=idx,
            start_time=start_time,
            end_time=start_time,
            duration_steps=1,
            params={
                "corruption_mode": mode,
                "original_value": current_val,
                "corrupted_value": float(corrupted_val),
            },
            description=f"Injected communication corruption ({desc}) on {var}."
        )

        return df_out, [event]
