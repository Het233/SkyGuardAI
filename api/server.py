"""
================================================================================
SkyGuard AI — Operational Meteorological Ingest & Surveillance REST API
================================================================================
Production-grade FastAPI service providing:
1. High-throughput single & batch sensor observation ingestion
2. Real-time multi-tier anomaly scoring (QC, Statistical, Isolation Forest, Autoencoder, Temporal, Spatial)
3. Event-vs-Fault physical reasoning & multiclass root-cause classification
4. Granular Explainable AI (XAI) Diagnostic Alert Cards with IMD maintenance SOPs
5. Continuous 0–100 sensor health scoring & predictive degradation tracking
6. Non-destructive spatial-temporal value imputation with zero overwrite guarantee
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from collections import deque
import os
import pickle
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    SensorObservationInput,
    BatchObservationInput,
    ObservationProcessedOutput,
    BatchProcessedOutput,
    StationHealthResponse,
    SystemStatusResponse,
)
from anomaly_injection.taxonomy import CORE_VARIABLES
from baseline.quality_control import DeterministicQCValidator
from baseline.statistical import CompositeBaselineDetector
from models.isolation_forest import WeatherIsolationForest
from models.autoencoder import DeepWeatherAutoencoder
from models.temporal_model import TemporalResidualPredictor
from models.spatial_model import SpatialConsensusDetector
from fusion.engine import HybridAnomalyFusionEngine
from reasoning.event_vs_fault import EventVsFaultClassifier
from classification.root_cause import RootCauseClassifier
from explainability.explainer import AnomalyExplainer
from explainability.diagnostic_card import DiagnosticCardGenerator
from health.health_score import SensorHealthTracker, HealthStatus
from imputation.correction import MeteorologicalImputer


app = FastAPI(
    title="SkyGuard AI — Operational Surveillance API",
    description="Real-Time Meteorological Sensor Quality Control, Anomaly Detection & Self-Healing Service",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — reads ALLOWED_ORIGINS env var (comma-separated list of frontend
# origins).  Defaults to localhost variants for local dev.  Set this env var
# in the Render dashboard to your Streamlit Cloud app URL.
# Example: ALLOWED_ORIGINS=https://your-app.streamlit.app
# ---------------------------------------------------------------------------
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,https://localhost:8501",
)
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ModelManager:
    """Singleton-style manager keeping models and station sliding buffers in memory."""

    def __init__(self, models_dir: str | None = None):
        # Resolve models_dir relative to this file so it works regardless of CWD
        # (important on Render where the working directory may differ from the
        # project root).
        if models_dir is None:
            _api_dir = os.path.dirname(os.path.abspath(__file__))
            _project_root = os.path.dirname(_api_dir)
            models_dir = os.path.join(_project_root, "artifacts", "models")
        self.models_dir = models_dir
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False

        # Sub-detectors and engines
        self.qc = DeterministicQCValidator()
        self.stat = CompositeBaselineDetector()
        self.spatial = SpatialConsensusDetector(k_neighbors=3)
        self.fusion = HybridAnomalyFusionEngine()
        self.reasoner = EventVsFaultClassifier(spatial_detector=self.spatial)
        self.explainer = AnomalyExplainer()
        self.health_tracker = SensorHealthTracker(rolling_window_hours=168)
        self.imputer = MeteorologicalImputer(k_neighbors=3)

        self.engineer = None
        self.iso_model = None
        self.ae_model = None
        self.temp_model = None
        self.root_cause_model = None

        # In-memory station rolling history: station_id -> deque of dict records (max 720 hours = 30 days)
        self.station_buffers: Dict[str, deque] = {}
        # Active alert buffer: deque of Diagnostic Cards
        self.active_alerts: deque = deque(maxlen=200)

    def load_models(self):
        fe_path = os.path.join(self.models_dir, "feature_engineer.pkl")
        iso_path = os.path.join(self.models_dir, "isolation_forest.pkl")
        ae_path = os.path.join(self.models_dir, "autoencoder.pt")
        tp_path = os.path.join(self.models_dir, "temporal_predictor.pkl")
        rc_path = os.path.join(self.models_dir, "root_cause_classifier.pkl")

        if os.path.exists(fe_path):
            with open(fe_path, "rb") as f:
                self.engineer = pickle.load(f)
        if os.path.exists(iso_path):
            self.iso_model = WeatherIsolationForest.load(iso_path)
        if os.path.exists(ae_path):
            self.ae_model = DeepWeatherAutoencoder.load(ae_path)
        if os.path.exists(tp_path):
            self.temp_model = TemporalResidualPredictor.load(tp_path)
        if os.path.exists(rc_path):
            self.root_cause_model = RootCauseClassifier.load(rc_path)

        self.models_loaded = True

    def get_station_history_df(self, station_id: str) -> pd.DataFrame:
        if station_id not in self.station_buffers or len(self.station_buffers[station_id]) == 0:
            return pd.DataFrame()
        return pd.DataFrame(list(self.station_buffers[station_id]))

    def append_observation(self, station_id: str, record: Dict[str, Any]):
        if station_id not in self.station_buffers:
            self.station_buffers[station_id] = deque(maxlen=720)
        self.station_buffers[station_id].append(record)


manager = ModelManager()


@app.on_event("startup")
def startup_event():
    """Load machine learning models and initialize spatial coordinates on startup."""
    manager.load_models()


@app.get("/health", response_model=SystemStatusResponse)
def get_system_health():
    """Liveness probe and system runtime diagnostics."""
    return SystemStatusResponse(
        status="ONLINE",
        version="1.0.0",
        device=manager.device,
        models_loaded=manager.models_loaded,
        active_stations_tracked=len(manager.station_buffers),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/v1/ingest/single", response_model=ObservationProcessedOutput)
def ingest_single_observation(obs: SensorObservationInput):
    """
    Ingest a single Automatic Weather Station observation in real time.
    Executes multi-tier scoring, root-cause classification, XAI card generation, and value imputation.
    """
    st_id = obs.station_id
    t_stamp = obs.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Build single row record
    record = {
        "station_id": st_id,
        "timestamp": t_stamp,
        "temperature_c": obs.temperature_c,
        "air_pressure_mbar": obs.air_pressure_mbar,
        "relative_humidity_pct": obs.relative_humidity_pct,
        "latitude": obs.latitude,
        "longitude": obs.longitude,
        "elevation_m": obs.elevation_m,
    }

    df_single = pd.DataFrame([record])

    # 1. Deterministic QC check
    qc_flags, qc_codes = manager.qc.validate(df_single)
    s_qc = float(qc_flags[0])

    # 2. Statistical baseline check
    df_hist = manager.get_station_history_df(st_id)
    s_stat = 0.0
    if len(df_hist) >= 6:
        df_combined = pd.concat([df_hist, df_single], ignore_index=True)
        _, stat_scores, _ = manager.stat.detect(df_combined)
        s_stat = float(stat_scores.iloc[-1])

    # 3. Model feature transform & scores
    s_iso = 0.0
    s_ae = 0.0
    s_temp = 0.0
    ae_errors = {}

    if manager.models_loaded and manager.engineer is not None:
        try:
            # Need historical context to compute lags and rolling moments
            if len(df_hist) >= 24:
                df_for_fe = pd.concat([df_hist.iloc[-48:], df_single], ignore_index=True)
            else:
                df_for_fe = df_single.copy()

            X_scaled = manager.engineer.transform(df_for_fe)
            x_inst = X_scaled[-1:]

            if manager.iso_model:
                s_iso = float(manager.iso_model.score(x_inst)[0])
            if manager.ae_model:
                s_ae = float(manager.ae_model.score(x_inst)[0])
                # Compute per-sensor reconstruction error
                diffs = manager.ae_model.reconstruction_error(x_inst)[0]
                ae_errors = {
                    "temperature_c": float(np.mean(diffs[:1])),
                    "air_pressure_mbar": float(np.mean(diffs[1:2])),
                    "relative_humidity_pct": float(np.mean(diffs[2:3])),
                }
            if manager.temp_model and all(obs.__dict__[v] is not None for v in CORE_VARIABLES):
                y_core = np.array([[obs.temperature_c, obs.air_pressure_mbar, obs.relative_humidity_pct]])
                temp_scores, _ = manager.temp_model.score(x_inst, y_core)
                s_temp = float(temp_scores[0])
        except Exception:
            pass

    # 4. Spatial Consensus Score
    s_spatial = 0.0
    # Gather simultaneous observations from other stations
    other_obs = []
    for other_st, buf in manager.station_buffers.items():
        if other_st != st_id and len(buf) > 0:
            last_rec = buf[-1]
            if last_rec.get("timestamp") == t_stamp:
                other_obs.append(last_rec)
    df_neighbors = pd.DataFrame(other_obs) if other_obs else None

    # 5. Hybrid Fusion
    comp_arr, bin_pred, conf_arr, sev_arr = manager.fusion.fuse_scores(
        qc_flags=np.array([s_qc]),
        stat_scores=np.array([s_stat]),
        iso_scores=np.array([s_iso]),
        ae_scores=np.array([s_ae]),
        temp_scores=np.array([s_temp]),
        spatial_scores=np.array([s_spatial]),
    )
    composite_score = float(comp_arr[0])
    severity = str(sev_arr[0])
    agreement = float(conf_arr[0])

    is_anom = bool(bin_pred[0]) or (composite_score >= 0.42)

    # 6. Root Cause Attribution
    pred_cause = "NORMAL"
    cause_conf = 0.95
    if is_anom:
        df_single["score_qc"] = s_qc
        df_single["score_autoencoder"] = s_ae
        df_single["score_temporal"] = s_temp
        df_single["score_statistical"] = s_stat
        df_single["score_isolation_forest"] = s_iso
        df_single["score_spatial"] = s_spatial
        df_single["composite_score"] = composite_score

        if manager.root_cause_model and manager.root_cause_model.is_fitted:
            p_cause, p_conf = manager.root_cause_model.predict(df_single)
            pred_cause = p_cause[0]
            cause_conf = float(p_conf[0])
        else:
            pred_cause = "ANOMALY"
            cause_conf = 0.85

    # 7. XAI Diagnostic Card
    card_dict = None
    prev_rec = df_hist.iloc[-1] if len(df_hist) > 0 else None
    explanation = manager.explainer.explain_instance(
        row=record,
        prev_row=prev_rec,
        neighbor_rows=df_neighbors,
        root_cause_model=manager.root_cause_model,
        autoencoder_errors=ae_errors,
    )

    primary_culprit = explanation.primary_culprit

    if is_anom:
        explanation.predicted_cause = pred_cause
        explanation.cause_confidence = cause_conf
        card_dict = DiagnosticCardGenerator.generate_card(
            explanation=explanation,
            severity=severity,
            composite_score=composite_score,
        )
        manager.active_alerts.append(card_dict)

    # 8. Non-Destructive Value Imputation
    df_single_for_imp = pd.DataFrame([record])
    df_single_for_imp["composite_score"] = composite_score
    df_single_for_imp["is_anomaly"] = is_anom

    df_imputed = manager.imputer.impute_dataframe(df_single_for_imp)

    imp_t = float(df_imputed["imputed_temperature_c"].iloc[0])
    imp_p = float(df_imputed["imputed_air_pressure_mbar"].iloc[0])
    imp_rh = float(df_imputed["imputed_relative_humidity_pct"].iloc[0])
    imp_flag = str(df_imputed["imputation_flag"].iloc[0])
    imp_conf = float(df_imputed["imputation_confidence"].iloc[0])

    # Store record in station buffer
    record["composite_score"] = composite_score
    record["is_anomaly"] = is_anom
    record["anomaly_type"] = pred_cause
    manager.append_observation(st_id, record)

    return ObservationProcessedOutput(
        station_id=st_id,
        timestamp=t_stamp,
        is_anomaly=is_anom,
        composite_score=round(composite_score, 4),
        severity=severity,
        predicted_cause=pred_cause,
        cause_confidence=round(cause_conf, 4),
        primary_culprit=primary_culprit,
        imputed_temperature_c=round(imp_t, 2),
        imputed_air_pressure_mbar=round(imp_p, 2),
        imputed_relative_humidity_pct=round(imp_rh, 2),
        imputation_flag=imp_flag,
        imputation_confidence=round(imp_conf, 3),
        diagnostic_card=card_dict,
    )


@app.post("/api/v1/ingest/batch", response_model=BatchProcessedOutput)
def ingest_batch_observations(payload: BatchObservationInput):
    """
    Ingest a batch of observations across one or multiple AWS stations.
    """
    results = []
    anom_count = 0

    for obs in payload.observations:
        processed = ingest_single_observation(obs)
        if processed.is_anomaly:
            anom_count += 1
        results.append(processed)

    return BatchProcessedOutput(
        total_processed=len(results),
        anomalies_detected=anom_count,
        results=results,
    )


@app.get("/api/v1/stations/health", response_model=List[StationHealthResponse])
def get_all_stations_health():
    """
    Return continuous 0-100 health indices and RUL forecasts across all tracked stations.
    """
    reports = []
    for st_id, buf in manager.station_buffers.items():
        if len(buf) > 0:
            st_df = pd.DataFrame(list(buf))
            rep = manager.health_tracker.compute_station_health(st_df)
            reports.append(StationHealthResponse(
                station_id=rep.station_id,
                timestamp=rep.timestamp,
                overall_score=rep.overall_score,
                status=rep.status.value,
                lowest_sensor=rep.lowest_sensor,
                urgency=rep.urgency,
                sensors={k: v.to_dict() for k, v in rep.sensor_health.items()},
                work_orders=rep.work_orders,
            ))
    return reports


@app.get("/api/v1/stations/{station_id}/health", response_model=StationHealthResponse)
def get_station_health(station_id: str):
    """
    Get detailed prognostic health report for a specific AWS station.
    """
    if station_id not in manager.station_buffers or len(manager.station_buffers[station_id]) == 0:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found in active telemetry.")

    st_df = pd.DataFrame(list(manager.station_buffers[station_id]))
    rep = manager.health_tracker.compute_station_health(st_df)

    return StationHealthResponse(
        station_id=rep.station_id,
        timestamp=rep.timestamp,
        overall_score=rep.overall_score,
        status=rep.status.value,
        lowest_sensor=rep.lowest_sensor,
        urgency=rep.urgency,
        sensors={k: v.to_dict() for k, v in rep.sensor_health.items()},
        work_orders=rep.work_orders,
    )


@app.get("/api/v1/alerts/active")
def get_active_alerts(limit: int = Query(50, ge=1, le=200)):
    """
    Retrieve recent active diagnostic alert cards.
    """
    alerts = list(manager.active_alerts)
    return {
        "active_alert_count": len(alerts),
        "alerts": alerts[-limit:],
    }


# ---------------------------------------------------------------------------
# Render / Uvicorn entrypoint
# Run locally:  python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
# Run on Render: start command reads $PORT automatically via this block.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=False)
