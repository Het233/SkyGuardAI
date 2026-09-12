"""
================================================================================
SkyGuard AI — Hybrid Anomaly Fusion Engine
================================================================================
Fuses signals across all 6 detection tiers:
  1. Deterministic QC (Physical bounds, step limits, persistence flatline, NaNs)
  2. Rolling Statistical Baselines (Z-score, IQR)
  3. Weather Isolation Forest (Unsupervised tree partition)
  4. Deep Weather Autoencoder (PyTorch manifold reconstruction)
  5. Temporal Residual Predictor (One-step-ahead forecasting residuals)
  6. Spatial Consensus Detector (Multi-station neighborhood consistency)

Produces:
  - Calibrated composite anomaly score in [0.0, 1.0]
  - Four-tier categorical severity: NORMAL, SUSPICIOUS, ANOMALY, CRITICAL
  - Inter-model consensus agreement confidence in [0.0, 1.0]
  - Detector score attribution decomposition
================================================================================
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from anomaly_injection.taxonomy import AnomalySeverity
from baseline.quality_control import DeterministicQCValidator
from baseline.statistical import CompositeBaselineDetector
from models.isolation_forest import WeatherIsolationForest
from models.autoencoder import DeepWeatherAutoencoder
from models.temporal_model import TemporalResidualPredictor
from models.spatial_model import SpatialConsensusDetector
from features.engineering import WeatherFeatureEngineer
from anomaly_injection.taxonomy import CORE_VARIABLES


class HybridAnomalyFusionEngine:
    """
    Central decision-making fusion engine for SkyGuard AI.
    """

    # Calibrated fusion weights
    DEFAULT_WEIGHTS = {
        "qc": 0.25,
        "autoencoder": 0.25,
        "temporal": 0.15,
        "statistical": 0.15,
        "spatial": 0.10,
        "isolation_forest": 0.10,
    }

    # Severity score cutoffs
    SEVERITY_THRESHOLDS = {
        "SUSPICIOUS": 0.35,
        "ANOMALY": 0.55,
        "CRITICAL": 0.75,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        decision_threshold: float = 0.45,
        include_spatial: bool = True,
    ):
        self.decision_threshold = decision_threshold
        self.include_spatial = include_spatial

        # Normalize weights
        raw_w = weights or self.DEFAULT_WEIGHTS.copy()
        if not self.include_spatial and "spatial" in raw_w:
            del raw_w["spatial"]
        total_w = sum(raw_w.values())
        self.weights = {k: v / total_w for k, v in raw_w.items()}

    def fuse_scores(
        self,
        qc_flags: np.ndarray,
        stat_scores: np.ndarray,
        iso_scores: np.ndarray,
        ae_scores: np.ndarray,
        temp_scores: np.ndarray,
        spatial_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Combine individual component scores into a calibrated composite score,
        binary prediction, confidence metric, and severity category.

        Returns:
            Tuple of:
                - composite_scores: Continuous anomaly score in [0.0, 1.0].
                - binary_predictions: Binary array [0 = normal, 1 = anomaly].
                - confidence_scores: Consensus agreement ratio in [0.0, 1.0].
                - severity_labels: Array of strings ("NORMAL", "SUSPICIOUS", "ANOMALY", "CRITICAL").
        """
        n = len(qc_flags)
        qc_arr = np.asarray(qc_flags).astype(float)
        stat_arr = np.asarray(stat_scores).astype(float)
        iso_arr = np.asarray(iso_scores).astype(float)
        ae_arr = np.asarray(ae_scores).astype(float)
        temp_arr = np.asarray(temp_scores).astype(float)

        if spatial_scores is not None and self.include_spatial:
            spat_arr = np.asarray(spatial_scores).astype(float)
        else:
            spat_arr = np.zeros(n)

        # Weighted linear combination
        w = self.weights
        composite = (
            w.get("qc", 0.0) * qc_arr
            + w.get("autoencoder", 0.0) * ae_arr
            + w.get("temporal", 0.0) * temp_arr
            + w.get("statistical", 0.0) * stat_arr
            + w.get("isolation_forest", 0.0) * iso_arr
            + w.get("spatial", 0.0) * spat_arr
        )

        # Deterministic hard override: physical impossibilities, flatlines, and NaNs trigger Critical
        composite = np.where(qc_arr >= 1.0, 1.0, composite)
        composite = np.clip(composite, 0.0, 1.0)

        # Binary decision
        binary_preds = (composite >= self.decision_threshold).astype(int)

        # Consensus Agreement Confidence: fraction of active detectors indicating anomalous behavior
        active_flags = [
            qc_arr >= 1.0,
            stat_arr >= 0.5,
            iso_arr >= 0.5,
            ae_arr >= 0.5,
            temp_arr >= 0.5,
        ]
        if self.include_spatial and spatial_scores is not None:
            active_flags.append(spat_arr >= 0.5)

        num_detectors = len(active_flags)
        agreeing_count = np.sum(active_flags, axis=0)
        confidence = agreeing_count / float(num_detectors)

        # Severity categorization
        severity = np.full(n, AnomalySeverity.NONE.value, dtype=object)
        severity[composite >= self.SEVERITY_THRESHOLDS["SUSPICIOUS"]] = AnomalySeverity.LOW.value
        severity[composite >= self.SEVERITY_THRESHOLDS["ANOMALY"]] = AnomalySeverity.HIGH.value
        severity[composite >= self.SEVERITY_THRESHOLDS["CRITICAL"]] = AnomalySeverity.CRITICAL.value

        return composite, binary_preds, confidence, severity

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        feature_engineer: WeatherFeatureEngineer,
        iso_model: WeatherIsolationForest,
        ae_model: DeepWeatherAutoencoder,
        tp_model: TemporalResidualPredictor,
        spatial_detector: Optional[SpatialConsensusDetector] = None,
    ) -> pd.DataFrame:
        """
        Execute full end-to-end multi-tier inference on an input DataFrame.

        Returns:
            pd.DataFrame containing original data plus composite anomaly scores,
            binary predictions, confidence, severity, and component breakdown.
        """
        df_out = df.copy()

        # Tier 1: Deterministic QC
        qc_validator = DeterministicQCValidator()
        qc_flags, _ = qc_validator.validate(df)

        # Tier 2: Statistical Baseline
        baseline_comp = CompositeBaselineDetector()
        _, stat_scores, _ = baseline_comp.detect(df)

        # Transform Features for ML models
        X_scaled = feature_engineer.transform(df)
        y_core = df[CORE_VARIABLES].values

        # Tier 3: Isolation Forest
        iso_scores = iso_model.score(X_scaled)

        # Tier 4: Deep Autoencoder
        ae_scores = ae_model.score(X_scaled)

        # Tier 5: Temporal Residuals
        temp_scores, _ = tp_model.score(X_scaled, y_core)

        # Tier 6: Spatial Consensus
        if self.include_spatial and spatial_detector is not None:
            _, spat_scores = spatial_detector.detect(df)
        else:
            spat_scores = pd.Series(0.0, index=df.index)

        # Fuse
        composite, preds, confidence, severity = self.fuse_scores(
            qc_flags=qc_flags.values,
            stat_scores=stat_scores.values,
            iso_scores=iso_scores,
            ae_scores=ae_scores,
            temp_scores=temp_scores,
            spatial_scores=spat_scores.values if self.include_spatial else None,
        )

        df_out["composite_score"] = np.round(composite, 4)
        df_out["predicted_anomaly"] = preds
        df_out["consensus_confidence"] = np.round(confidence, 3)
        df_out["predicted_severity"] = severity

        # Component attributions
        df_out["score_qc"] = np.round(qc_flags.astype(float).values, 2)
        df_out["score_autoencoder"] = np.round(ae_scores, 4)
        df_out["score_temporal"] = np.round(temp_scores, 4)
        df_out["score_statistical"] = np.round(stat_scores.values, 4)
        df_out["score_isolation_forest"] = np.round(iso_scores, 4)
        if self.include_spatial:
            df_out["score_spatial"] = np.round(spat_scores.values, 4)

        return df_out
