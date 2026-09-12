"""
================================================================================
SkyGuard AI — Master Synthetic Anomaly Generator
================================================================================
Orchestrates the injection of heterogeneous, realistic sensor and telemetry faults
into AWS weather datasets. Produces comprehensive ground-truth annotations and
preserves pristine signals for downstream evaluation and imputation benchmarking.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent, CORE_VARIABLES
from .spikes import SpikeInjector
from .freeze import FrozenSensorInjector
from .drift import DriftInjector, ConstantOffsetInjector
from .noise import NoiseInjector
from .missing import MissingDataInjector
from .communication import CommunicationCorruptionInjector
from .physical import PhysicalBoundaryInjector
from .multivariate import CrossSensorInconsistencyInjector, CoordinatedAnomalyInjector


# Default balanced distribution weights across anomaly types
DEFAULT_ANOMALY_WEIGHTS: Dict[AnomalyType, float] = {
    AnomalyType.SPIKE: 0.15,
    AnomalyType.DROP: 0.10,
    AnomalyType.FROZEN_SENSOR: 0.15,
    AnomalyType.SENSOR_DRIFT: 0.15,
    AnomalyType.CONSTANT_OFFSET: 0.10,
    AnomalyType.HIGH_NOISE: 0.10,
    AnomalyType.MISSING_DATA: 0.08,
    AnomalyType.COMMUNICATION_CORRUPTION: 0.05,
    AnomalyType.PHYSICALLY_IMPOSSIBLE: 0.05,
    AnomalyType.CROSS_SENSOR_INCONSISTENCY: 0.04,
    AnomalyType.COORDINATED_MULTIVARIATE: 0.03,
}


class SyntheticAnomalyGenerator:
    """
    Master generator that injects a curated mix of synthetic faults into AWS observations.
    """

    def __init__(
        self,
        anomaly_rate: float = 0.05,
        weights: Optional[Dict[AnomalyType, float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the generator.

        Args:
            anomaly_rate: Overall target fraction of rows to corrupt (e.g. 0.05 = 5%).
            weights: Optional dictionary mapping AnomalyType to relative probability weight.
            seed: Optional integer seed for reproducibility.
        """
        if not 0.0 < anomaly_rate < 0.5:
            raise ValueError(f"anomaly_rate must be between 0.0 and 0.5, got {anomaly_rate}")

        self.anomaly_rate = anomaly_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Normalize weights
        raw_weights = weights if weights is not None else DEFAULT_ANOMALY_WEIGHTS
        total_w = sum(raw_weights.values())
        self.weights = {k: v / total_w for k, v in raw_weights.items()}

        # Instantiate sub-injectors with child seeds
        self.injectors = {
            AnomalyType.SPIKE: SpikeInjector(seed=self._next_seed()),
            AnomalyType.DROP: SpikeInjector(seed=self._next_seed()),
            AnomalyType.FROZEN_SENSOR: FrozenSensorInjector(seed=self._next_seed()),
            AnomalyType.SENSOR_DRIFT: DriftInjector(seed=self._next_seed()),
            AnomalyType.CONSTANT_OFFSET: ConstantOffsetInjector(seed=self._next_seed()),
            AnomalyType.HIGH_NOISE: NoiseInjector(seed=self._next_seed()),
            AnomalyType.MISSING_DATA: MissingDataInjector(seed=self._next_seed()),
            AnomalyType.COMMUNICATION_CORRUPTION: CommunicationCorruptionInjector(seed=self._next_seed()),
            AnomalyType.PHYSICALLY_IMPOSSIBLE: PhysicalBoundaryInjector(seed=self._next_seed()),
            AnomalyType.CROSS_SENSOR_INCONSISTENCY: CrossSensorInconsistencyInjector(seed=self._next_seed()),
            AnomalyType.COORDINATED_MULTIVARIATE: CoordinatedAnomalyInjector(seed=self._next_seed()),
        }

        self.events: List[AnomalyEvent] = []

    def _next_seed(self) -> int:
        """Generate deterministic child seeds."""
        return int(self.rng.randint(0, 2**31 - 1))

    def generate(
        self,
        df: pd.DataFrame,
        station_subset: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Generate synthetic anomalies across the provided DataFrame.

        Args:
            df: Input clean AWS observations.
            station_subset: Optional subset of station IDs to process.

        Returns:
            Tuple of:
                - Augmented DataFrame containing ground truth columns and pristine backup columns.
                - List of AnomalyEvent records.
        """
        for col in CORE_VARIABLES:
            if col not in df.columns:
                raise ValueError(f"Missing core variable '{col}' in input DataFrame.")
        if "station_id" not in df.columns:
            raise ValueError("Input DataFrame missing 'station_id' column.")

        df_out = df.copy()

        # Preserve pristine original data for error residual / imputation evaluation
        for var in CORE_VARIABLES:
            df_out[f"clean_{var}"] = df_out[var].copy()

        # Initialize ground truth label columns
        df_out["is_anomaly"] = 0
        df_out["anomaly_type"] = AnomalyType.NORMAL.value
        df_out["anomaly_severity"] = AnomalySeverity.NONE.value
        df_out["anomaly_target"] = "none"

        stations = df_out["station_id"].unique()
        if station_subset is not None:
            stations = [s for s in stations if s in station_subset]

        all_events: List[AnomalyEvent] = []

        type_keys = list(self.weights.keys())
        prob_vals = [self.weights[k] for k in type_keys]

        for st in stations:
            st_indices = df_out.index[df_out["station_id"] == st].tolist()
            n_rows = len(st_indices)
            if n_rows < 10:
                continue

            target_anomaly_rows = int(n_rows * self.anomaly_rate)
            injected_rows = 0
            occupied_indices = set()

            # Attempt injections until target anomaly row budget is reached or max trials exceeded
            trials = 0
            max_trials = target_anomaly_rows * 5

            while injected_rows < target_anomaly_rows and trials < max_trials:
                trials += 1
                selected_idx = int(self.rng.choice(len(type_keys), p=prob_vals))
                selected_type = type_keys[selected_idx]

                # Determine candidate duration
                if selected_type in (AnomalyType.SPIKE, AnomalyType.DROP, AnomalyType.PHYSICALLY_IMPOSSIBLE, AnomalyType.COMMUNICATION_CORRUPTION):
                    duration = 1
                elif selected_type in (AnomalyType.CROSS_SENSOR_INCONSISTENCY, AnomalyType.COORDINATED_MULTIVARIATE):
                    duration = int(self.rng.randint(1, 4))
                elif selected_type == AnomalyType.MISSING_DATA:
                    duration = int(self.rng.randint(2, 10))
                elif selected_type == AnomalyType.HIGH_NOISE:
                    duration = int(self.rng.randint(6, 24))
                elif selected_type == AnomalyType.FROZEN_SENSOR:
                    duration = int(self.rng.randint(6, 24))
                elif selected_type == AnomalyType.CONSTANT_OFFSET:
                    duration = int(self.rng.randint(12, 36))
                elif selected_type == AnomalyType.SENSOR_DRIFT:
                    duration = int(self.rng.randint(18, 48))
                else:
                    duration = 1

                if duration >= n_rows - 2:
                    continue

                # Find an unoccupied starting index within this station
                max_start = n_rows - duration - 1
                rel_start = int(self.rng.randint(1, max(2, max_start)))
                candidate_indices = set(st_indices[rel_start + i] for i in range(duration))

                # Avoid overlap with existing anomalies for clean ground truth
                if candidate_indices & occupied_indices:
                    continue

                # Execute injection
                injector = self.injectors[selected_type]
                sub_kwargs = {"station_id": st, "start_idx": rel_start, "duration": duration}

                if selected_type == AnomalyType.SPIKE:
                    sub_kwargs["direction"] = "positive"
                elif selected_type == AnomalyType.DROP:
                    sub_kwargs["direction"] = "negative"

                try:
                    df_out, ev_list = injector.inject(df_out, **sub_kwargs)
                except Exception:
                    continue

                if not ev_list:
                    continue

                ev = ev_list[0]
                all_events.append(ev)

                # Update ground-truth columns for the affected indices
                ev_indices = [st_indices[rel_start + i] for i in range(duration)]
                df_out.loc[ev_indices, "is_anomaly"] = 1
                df_out.loc[ev_indices, "anomaly_type"] = ev.anomaly_type.value
                df_out.loc[ev_indices, "anomaly_severity"] = ev.severity.value
                df_out.loc[ev_indices, "anomaly_target"] = ",".join(ev.target_variables)

                occupied_indices.update(candidate_indices)
                injected_rows += duration

        self.events = all_events
        return df_out, all_events

    def get_summary_report(self, df_augmented: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a statistical summary report of the synthetic injection run.
        """
        total_rows = len(df_augmented)
        total_anomalies = int(df_augmented["is_anomaly"].sum())
        anomaly_pct = (total_anomalies / total_rows * 100.0) if total_rows > 0 else 0.0

        type_counts = df_augmented["anomaly_type"].value_counts().to_dict()
        severity_counts = df_augmented["anomaly_severity"].value_counts().to_dict()

        return {
            "total_observations": total_rows,
            "total_anomalous_rows": total_anomalies,
            "actual_anomaly_rate_pct": round(anomaly_pct, 2),
            "total_events_injected": len(self.events),
            "distribution_by_type": type_counts,
            "distribution_by_severity": severity_counts,
        }

    def save_event_registry(self, filepath: str) -> None:
        """Save ground-truth event registry to a JSON file."""
        records = [ev.to_dict() for ev in self.events]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
