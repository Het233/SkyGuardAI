"""
================================================================================
SkyGuard AI — Anomaly Taxonomy & Metadata Definitions
================================================================================
Defines standardized enumerations and data structures for all meteorological
sensor anomaly classes, severity levels, and ground-truth metadata.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


class AnomalyType(str, Enum):
    """Enumeration of all supported AWS sensor anomaly types."""
    NORMAL = "NORMAL"
    SPIKE = "SPIKE"
    DROP = "DROP"
    FROZEN_SENSOR = "FROZEN_SENSOR"
    SENSOR_DRIFT = "SENSOR_DRIFT"
    CONSTANT_OFFSET = "CONSTANT_OFFSET"
    HIGH_NOISE = "HIGH_NOISE"
    MISSING_DATA = "MISSING_DATA"
    COMMUNICATION_CORRUPTION = "COMMUNICATION_CORRUPTION"
    PHYSICALLY_IMPOSSIBLE = "PHYSICALLY_IMPOSSIBLE"
    CROSS_SENSOR_INCONSISTENCY = "CROSS_SENSOR_INCONSISTENCY"
    COORDINATED_MULTIVARIATE = "COORDINATED_MULTIVARIATE"


class AnomalySeverity(str, Enum):
    """Categorical severity classification for detected or injected anomalies."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Core sensor variables allowed under SkyGuard AI physical constraints
CORE_VARIABLES: List[str] = [
    "temperature_c",
    "air_pressure_mbar",
    "relative_humidity_pct",
]


@dataclass
class AnomalyEvent:
    """Ground truth metadata record for a single injected anomaly event."""
    event_id: str
    station_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    target_variables: List[str]
    start_idx: int
    end_idx: int
    start_time: str
    end_time: str
    duration_steps: int
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert event record to dictionary representation."""
        data = asdict(self)
        data["anomaly_type"] = self.anomaly_type.value
        data["severity"] = self.severity.value
        return data
