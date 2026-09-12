"""
================================================================================
SkyGuard AI — Sensor Health Scoring & Predictive Degradation Tracker
================================================================================
Maintains continuous 0–100 health indices per meteorological sensor and station,
tracks cumulative hardware degradation, projects Remaining Useful Life (RUL),
and issues predictive maintenance work orders before catastrophic sensor failure.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from anomaly_injection.taxonomy import CORE_VARIABLES


class HealthStatus(str, Enum):
    EXCELLENT = "EXCELLENT"     # 90.0 - 100.0
    GOOD = "GOOD"               # 75.0 - 89.9
    DEGRADING = "DEGRADING"     # 50.0 - 74.9
    CRITICAL = "CRITICAL"       # < 50.0


@dataclass
class SensorHealthIndex:
    """Detailed health record for a single meteorological sensor."""
    variable: str
    score: float
    status: HealthStatus
    anomaly_rate: float
    drift_magnitude: float
    noise_index: float
    flatline_count: int
    missing_rate: float
    decay_rate_per_day: float
    days_to_critical: Optional[float]
    maintenance_advice: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "score": round(self.score, 1),
            "status": self.status.value,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "drift_magnitude": round(self.drift_magnitude, 2),
            "noise_index": round(self.noise_index, 2),
            "flatline_count": int(self.flatline_count),
            "missing_rate": round(self.missing_rate, 4),
            "decay_rate_per_day": round(self.decay_rate_per_day, 2),
            "days_to_critical": round(self.days_to_critical, 1) if self.days_to_critical is not None else None,
            "maintenance_advice": self.maintenance_advice,
        }


@dataclass
class StationHealthReport:
    """Comprehensive health summary for an entire Automatic Weather Station."""
    station_id: str
    timestamp: str
    overall_score: float
    status: HealthStatus
    sensor_health: Dict[str, SensorHealthIndex]
    lowest_sensor: str
    urgency: str
    work_orders: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "timestamp": str(self.timestamp),
            "overall_score": round(self.overall_score, 1),
            "status": self.status.value,
            "lowest_sensor": self.lowest_sensor,
            "urgency": self.urgency,
            "sensors": {k: v.to_dict() for k, v in self.sensor_health.items()},
            "work_orders": self.work_orders,
        }


class SensorHealthTracker:
    """
    Computes rolling degradation metrics and prognostic health indices for AWS sensors.
    """

    # Baseline noise thresholds (standard deviation of high-pass filter residuals)
    NOMINAL_NOISE = {
        "temperature_c": 0.4,       # °C
        "air_pressure_mbar": 0.3,   # mbar
        "relative_humidity_pct": 1.2 # %
    }

    def __init__(self, rolling_window_hours: int = 168):
        """
        Initialize the tracker.
        
        Args:
            rolling_window_hours: Window length in hours for degradation tracking (default 168h = 7 days).
        """
        self.rolling_window_hours = rolling_window_hours

    def compute_station_health(
        self,
        station_df: pd.DataFrame,
        anomaly_flags: Optional[Union[pd.Series, np.ndarray]] = None,
        anomaly_types: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> StationHealthReport:
        """
        Compute health indices across all 3 sensors and overall station health.

        Args:
            station_df: Time-series DataFrame for a single station, sorted by timestamp.
            anomaly_flags: Optional boolean or 0/1 array flagging anomalous timestamps.
            anomaly_types: Optional array of string anomaly types for fine-grained penalty.
        """
        if len(station_df) == 0:
            raise ValueError("Input DataFrame is empty.")

        station_id = str(station_df["station_id"].iloc[0]) if "station_id" in station_df.columns else "AWS_UNKNOWN"
        timestamp = str(station_df["timestamp"].iloc[-1]) if "timestamp" in station_df.columns else "CURRENT"

        # Align anomaly flags
        if anomaly_flags is None:
            if "is_anomaly" in station_df.columns:
                flags = station_df["is_anomaly"].values.astype(bool)
            elif "composite_score" in station_df.columns:
                flags = (station_df["composite_score"].values >= 0.42)
            else:
                flags = np.zeros(len(station_df), dtype=bool)
        else:
            flags = np.asarray(anomaly_flags, dtype=bool)

        if anomaly_types is None:
            if "anomaly_type" in station_df.columns:
                types = station_df["anomaly_type"].values.astype(str)
            else:
                types = np.array(["NORMAL"] * len(station_df))
        else:
            types = np.asarray(anomaly_types, dtype=str)

        # Truncate to evaluation window
        win_size = min(len(station_df), self.rolling_window_hours)
        df_win = station_df.iloc[-win_size:].copy()
        flags_win = flags[-win_size:]
        types_win = types[-win_size:]

        sensor_records: Dict[str, SensorHealthIndex] = {}
        work_orders: List[str] = []

        for var in CORE_VARIABLES:
            rec = self._compute_single_sensor_health(
                series=df_win[var] if var in df_win.columns else pd.Series(dtype=float),
                var_name=var,
                flags=flags_win,
                types=types_win,
                total_hours=win_size,
            )
            sensor_records[var] = rec
            if rec.status in [HealthStatus.DEGRADING, HealthStatus.CRITICAL]:
                work_orders.append(
                    f"[{rec.status.value}] {var.upper()}: {rec.maintenance_advice} "
                    f"(Score: {rec.score:.1f}/100, RUL: {f'{rec.days_to_critical:.1f} days' if rec.days_to_critical else 'Imminent'})"
                )

        # Aggregate station overall score
        # Bottleneck-sensitive: 50% min score + 50% mean score
        scores = [rec.score for rec in sensor_records.values()]
        min_score = min(scores)
        mean_score = float(np.mean(scores))
        overall_score = float(np.clip(0.5 * min_score + 0.5 * mean_score, 0.0, 100.0))

        overall_status = self._score_to_status(overall_score)
        lowest_sensor = min(sensor_records.keys(), key=lambda k: sensor_records[k].score)

        urgency = "ROUTINE"
        if overall_status == HealthStatus.CRITICAL:
            urgency = "EMERGENCY_DISPATCH"
        elif overall_status == HealthStatus.DEGRADING:
            urgency = "SCHEDULED_MAINTENANCE"

        return StationHealthReport(
            station_id=station_id,
            timestamp=timestamp,
            overall_score=overall_score,
            status=overall_status,
            sensor_health=sensor_records,
            lowest_sensor=lowest_sensor,
            urgency=urgency,
            work_orders=work_orders,
        )

    def _compute_single_sensor_health(
        self,
        series: pd.Series,
        var_name: str,
        flags: np.ndarray,
        types: np.ndarray,
        total_hours: int,
    ) -> SensorHealthIndex:
        """Calculate health index and prognostic degradation for one sensor."""
        if len(series) == 0 or series.isna().all():
            return SensorHealthIndex(
                variable=var_name,
                score=0.0,
                status=HealthStatus.CRITICAL,
                anomaly_rate=1.0,
                drift_magnitude=0.0,
                noise_index=0.0,
                flatline_count=0,
                missing_rate=1.0,
                decay_rate_per_day=0.0,
                days_to_critical=0.0,
                maintenance_advice="Sensor offline or channel unassigned. Reconnect sensor probe.",
            )

        # 1. Missing Rate
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / max(total_hours, 1)

        # Fill missing for time-series feature evaluation
        clean_s = series.ffill().bfill().fillna(0.0)

        # 2. Flatline Count (persistence of identical values > 3 steps)
        diffs = clean_s.diff().abs()
        is_flat = (diffs < 1e-4).astype(int)
        flat_runs = (is_flat * (is_flat.groupby((is_flat != is_flat.shift()).cumsum()).cumcount() + 1))
        flatline_count = int((flat_runs >= 4).sum())

        # 3. Noise Ratio (high-frequency second differences vs nominal)
        sec_diffs = clean_s.diff().diff().dropna()
        emp_noise = float(sec_diffs.std()) if len(sec_diffs) > 1 else 0.0
        nominal_noise = self.NOMINAL_NOISE.get(var_name, 1.0)
        noise_ratio = emp_noise / nominal_noise

        # 4. Drift Magnitude (low-frequency trend slope or monotonic offset)
        x_axis = np.arange(len(clean_s))
        if len(clean_s) >= 12:
            p = np.polyfit(x_axis, clean_s.values, 1)
            drift_slope_per_hr = abs(float(p[0]))
            drift_magnitude = drift_slope_per_hr * 24.0  # expected drift per 24 hours
        else:
            drift_magnitude = 0.0

        # 5. Anomaly Rate & Failure Penalties
        anomaly_count = int(flags.sum())
        anomaly_rate = anomaly_count / max(total_hours, 1)

        # Deductions calculation starting from pristine 100.0
        score = 100.0

        # Anomaly rate penalty: 5% anomalies deducts 20 points; 20% deducts 60 points
        score -= min(anomaly_rate * 300.0, 50.0)

        # Missing data penalty
        score -= min(missing_rate * 150.0, 40.0)

        # Flatline penalty
        if flatline_count > 0:
            score -= min(flatline_count * 3.0, 30.0)

        # Noise penalty (penalize if noise > 2.5x nominal)
        if noise_ratio > 2.5:
            score -= min((noise_ratio - 2.5) * 8.0, 25.0)

        # Drift penalty
        drift_limit = {"temperature_c": 1.5, "air_pressure_mbar": 2.0, "relative_humidity_pct": 5.0}.get(var_name, 2.0)
        if drift_magnitude > drift_limit:
            score -= min((drift_magnitude - drift_limit) * 10.0, 30.0)

        score = float(np.clip(score, 0.0, 100.0))
        status = self._score_to_status(score)

        # 6. Prognostic Health Trajectory & RUL (Days to Critical)
        # Partition window into 4 sequential sub-windows to estimate degradation rate
        decay_rate_per_day = 0.0
        days_to_critical = None

        if len(clean_s) >= 48:
            step = len(clean_s) // 4
            sub_scores = []
            for i in range(4):
                idx_start = i * step
                idx_end = (i + 1) * step if i < 3 else len(clean_s)
                sub_flags = flags[idx_start:idx_end]
                sub_rate = sub_flags.sum() / max(len(sub_flags), 1)
                sub_score = 100.0 - min(sub_rate * 300.0, 50.0)
                sub_scores.append(sub_score)

            t_days = np.array([0.0, 0.25, 0.5, 1.0]) * (len(clean_s) / 24.0)
            slope = float(np.polyfit(t_days, sub_scores, 1)[0])
            decay_rate_per_day = -slope  # positive value = declining health

            if decay_rate_per_day > 0.5 and score > 50.0:
                days_to_critical = (score - 50.0) / decay_rate_per_day
            elif score <= 50.0:
                days_to_critical = 0.0

        # Formulate targeted maintenance advice
        advice = self._generate_advice(var_name, status, flatline_count, noise_ratio, drift_magnitude, missing_rate)

        return SensorHealthIndex(
            variable=var_name,
            score=score,
            status=status,
            anomaly_rate=anomaly_rate,
            drift_magnitude=drift_magnitude,
            noise_index=noise_ratio,
            flatline_count=flatline_count,
            missing_rate=missing_rate,
            decay_rate_per_day=decay_rate_per_day,
            days_to_critical=days_to_critical,
            maintenance_advice=advice,
        )

    @staticmethod
    def _score_to_status(score: float) -> HealthStatus:
        if score >= 90.0:
            return HealthStatus.EXCELLENT
        elif score >= 75.0:
            return HealthStatus.GOOD
        elif score >= 50.0:
            return HealthStatus.DEGRADING
        else:
            return HealthStatus.CRITICAL

    @staticmethod
    def _generate_advice(
        var: str,
        status: HealthStatus,
        flatline_cnt: int,
        noise_ratio: float,
        drift_mag: float,
        missing_rate: float,
    ) -> str:
        if status == HealthStatus.EXCELLENT:
            return "Sensor functioning nominally. Continue routine monitoring."
        if status == HealthStatus.GOOD:
            return "Minor signal fluctuations detected. Observe over next 72-hour cycle."

        # Degrading or Critical diagnoses
        if missing_rate > 0.15:
            return f"Severe packet loss ({missing_rate*100:.1f}%). Check {var} cable connection and logger port."
        if flatline_cnt > 3:
            return f"Persistent signal lockup detected ({flatline_cnt} freeze frames). Power-cycle logger and test transducer bus."
        if noise_ratio > 3.0:
            return f"Excessive high-frequency noise ({noise_ratio:.1f}x nominal). Check grounding rod and shielded cable braid."
        if drift_mag > 2.0:
            return f"Monotonic calibration drift detected (~{drift_mag:.1f} units/day). Schedule sensor recalibration."

        return f"Multiple degradation signatures detected on {var}. Conduct comprehensive physical inspection."
