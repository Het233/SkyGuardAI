"""
================================================================================
SkyGuard AI — Base Anomaly Injector
================================================================================
Abstract base class and shared utilities for all synthetic anomaly injectors.
Enforces station isolation, temporal consistency, and ground-truth metadata logging.
"""

from abc import ABC, abstractmethod
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .taxonomy import AnomalyType, AnomalySeverity, AnomalyEvent, CORE_VARIABLES


class BaseAnomalyInjector(ABC):
    """
    Abstract base class for all synthetic anomaly injectors in SkyGuard AI.
    All subclasses must implement the `inject` method.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize base injector with an optional random seed for reproducibility.

        Args:
            seed: Integer random seed for reproducible injection.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed: int) -> None:
        """Reset internal random generator state."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def inject(
        self,
        df: pd.DataFrame,
        target_variable: Optional[str] = None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[AnomalyEvent]]:
        """
        Inject anomalies into the provided DataFrame.

        Args:
            df: Input DataFrame containing AWS observations.
            target_variable: Specific column to perturb. If None, chosen randomly.
            **kwargs: Injector-specific hyper-parameters.

        Returns:
            Tuple of:
                - Modified DataFrame containing injected values and annotations.
                - List of AnomalyEvent records documenting ground-truth metadata.
        """
        pass

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        """Verify required columns exist in input DataFrame."""
        missing = [col for col in CORE_VARIABLES if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required core variables: {missing}")
        if "station_id" not in df.columns:
            raise ValueError("Input DataFrame missing 'station_id' column required for station isolation.")

    def _select_variable(self, target_variable: Optional[str]) -> str:
        """Validate or randomly select one of the core variables."""
        if target_variable is not None:
            if target_variable not in CORE_VARIABLES:
                raise ValueError(
                    f"Target variable '{target_variable}' is not allowed. "
                    f"Must be one of {CORE_VARIABLES} to satisfy physical constraints."
                )
            return target_variable
        return self.rng.choice(CORE_VARIABLES)

    def _generate_event_id(self) -> str:
        """Generate a short unique event identifier."""
        return f"EVT_{uuid.uuid4().hex[:8].upper()}"

    def _determine_severity(self, deviation_ratio: float) -> AnomalySeverity:
        """
        Classify anomaly severity based on the relative magnitude of the deviation.
        deviation_ratio is typically (deviation / variable_std).
        """
        abs_ratio = abs(deviation_ratio)
        if abs_ratio < 2.0:
            return AnomalySeverity.LOW
        elif abs_ratio < 4.0:
            return AnomalySeverity.MEDIUM
        elif abs_ratio < 6.0:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL
