"""
================================================================================
SkyGuard AI — Anomaly Injection Package
================================================================================
"""

from .taxonomy import (
    AnomalyType,
    AnomalySeverity,
    AnomalyEvent,
    CORE_VARIABLES,
)
from .base import BaseAnomalyInjector
from .spikes import SpikeInjector
from .freeze import FrozenSensorInjector
from .drift import DriftInjector, ConstantOffsetInjector
from .noise import NoiseInjector
from .missing import MissingDataInjector
from .communication import CommunicationCorruptionInjector
from .physical import PhysicalBoundaryInjector
from .multivariate import (
    CrossSensorInconsistencyInjector,
    CoordinatedAnomalyInjector,
)
from .generator import SyntheticAnomalyGenerator, DEFAULT_ANOMALY_WEIGHTS

__all__ = [
    "AnomalyType",
    "AnomalySeverity",
    "AnomalyEvent",
    "CORE_VARIABLES",
    "BaseAnomalyInjector",
    "SpikeInjector",
    "FrozenSensorInjector",
    "DriftInjector",
    "ConstantOffsetInjector",
    "NoiseInjector",
    "MissingDataInjector",
    "CommunicationCorruptionInjector",
    "PhysicalBoundaryInjector",
    "CrossSensorInconsistencyInjector",
    "CoordinatedAnomalyInjector",
    "SyntheticAnomalyGenerator",
    "DEFAULT_ANOMALY_WEIGHTS",
]
