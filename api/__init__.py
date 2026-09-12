"""
SkyGuard AI — REST API Package
FastAPI-based streaming ingest, anomaly scoring, XAI alert cards, and health monitoring.
"""

from .schemas import (
    SensorObservationInput,
    BatchObservationInput,
    ObservationProcessedOutput,
    StationHealthResponse,
)
from .server import app

__all__ = [
    "SensorObservationInput",
    "BatchObservationInput",
    "ObservationProcessedOutput",
    "StationHealthResponse",
    "app",
]
