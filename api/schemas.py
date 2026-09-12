"""
================================================================================
SkyGuard AI — Pydantic Schemas for Streaming REST API
================================================================================
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class SensorObservationInput(BaseModel):
    """Input payload for a single Automatic Weather Station observation."""
    station_id: str = Field(..., example="IMD_AWS_0001", description="Unique station alphanumeric identifier")
    timestamp: Optional[str] = Field(None, example="2024-01-01T12:00:00", description="Observation timestamp in ISO 8601")
    temperature_c: Optional[float] = Field(None, example=28.4, description="Ambient air temperature in degrees Celsius")
    air_pressure_mbar: Optional[float] = Field(None, example=1008.2, description="Surface barometric pressure in mbar/hPa")
    relative_humidity_pct: Optional[float] = Field(None, example=64.0, description="Relative humidity in percentage (0-100%)")
    latitude: Optional[float] = Field(None, example=12.9716, description="Station geographic latitude")
    longitude: Optional[float] = Field(None, example=77.5946, description="Station geographic longitude")
    elevation_m: Optional[float] = Field(None, example=920.0, description="Station elevation above mean sea level in meters")


class BatchObservationInput(BaseModel):
    """Input payload for multiple simultaneous AWS observations."""
    observations: List[SensorObservationInput]


class ObservationProcessedOutput(BaseModel):
    """Processed output containing detection, diagnosis, XAI card, and imputed values."""
    station_id: str
    timestamp: str
    is_anomaly: bool
    composite_score: float
    severity: str
    predicted_cause: str
    cause_confidence: float
    primary_culprit: str
    imputed_temperature_c: float
    imputed_air_pressure_mbar: float
    imputed_relative_humidity_pct: float
    imputation_flag: str
    imputation_confidence: float
    diagnostic_card: Optional[Dict[str, Any]] = None


class BatchProcessedOutput(BaseModel):
    """Response containing processed records and summary metadata."""
    total_processed: int
    anomalies_detected: int
    results: List[ObservationProcessedOutput]


class StationHealthResponse(BaseModel):
    """Prognostic health report for a weather station."""
    station_id: str
    timestamp: str
    overall_score: float
    status: str
    lowest_sensor: str
    urgency: str
    sensors: Dict[str, Any]
    work_orders: List[str]


class SystemStatusResponse(BaseModel):
    """System health and operational status response."""
    status: str
    version: str = "1.0.0"
    device: str
    models_loaded: bool
    active_stations_tracked: int
    timestamp: str
