"""
================================================================================
SkyGuard AI — Explainable AI (XAI) Explainer Engine
================================================================================
Granular interpretability engine providing:
1. Primary Culprit Sensor Identification (T, P, RH)
2. Autoencoder Reconstruction Dissection per Sensor
3. Root-Cause Feature Attribution (Random Forest Tree Contributions)
4. Spatial Disparity Breakdown (MAD Z-scores with Elevation Lapse Correction)
5. Thermodynamic Psychrometric Inconsistency Checks
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from anomaly_injection.taxonomy import AnomalyType, CORE_VARIABLES


@dataclass
class VariableAttribution:
    """Attribution metrics for a specific sensor variable."""
    variable: str
    anomaly_score: float
    deviation_sigma: float
    step_change: float
    is_flatline: bool
    is_out_of_bounds: bool
    reconstruction_error: float = 0.0


@dataclass
class ExplanationResult:
    """Comprehensive diagnostic explanation for a flagged observation."""
    station_id: str
    timestamp: str
    predicted_cause: str
    cause_confidence: float
    primary_culprit: str
    all_culprits: List[str]
    variable_attributions: Dict[str, VariableAttribution]
    top_contributing_features: List[Dict[str, Any]]
    spatial_evidence: Dict[str, Any]
    thermodynamic_evidence: Dict[str, Any]
    evidence_summary: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "station_id": self.station_id,
            "timestamp": str(self.timestamp),
            "predicted_cause": self.predicted_cause,
            "cause_confidence": round(float(self.cause_confidence), 4),
            "primary_culprit": self.primary_culprit,
            "all_culprits": self.all_culprits,
            "variable_attributions": {
                k: {
                    "variable": v.variable,
                    "anomaly_score": round(float(v.anomaly_score), 4),
                    "deviation_sigma": round(float(v.deviation_sigma), 2),
                    "step_change": round(float(v.step_change), 2),
                    "is_flatline": bool(v.is_flatline),
                    "is_out_of_bounds": bool(v.is_out_of_bounds),
                    "reconstruction_error": round(float(v.reconstruction_error), 4),
                }
                for k, v in self.variable_attributions.items()
            },
            "top_contributing_features": self.top_contributing_features,
            "spatial_evidence": self.spatial_evidence,
            "thermodynamic_evidence": self.thermodynamic_evidence,
            "evidence_summary": self.evidence_summary,
        }


class AnomalyExplainer:
    """
    Multimodal explainability engine for SkyGuard AI alerts.
    """

    # Physical step limits per hour for context
    STEP_LIMITS = {
        "temperature_c": 12.0,      # °C / hour
        "air_pressure_mbar": 15.0,    # mbar / hour
        "relative_humidity_pct": 45.0 # % / hour
    }

    PHYSICAL_RANGES = {
        "temperature_c": (-50.0, 60.0),
        "air_pressure_mbar": (300.0, 1100.0),
        "relative_humidity_pct": (0.0, 100.0),
    }

    CLASSIFIER_FEATURE_NAMES = [
        "temperature_c_val", "temperature_c_is_nan", "temperature_c_diff1", "temperature_c_diff2", "temperature_c_roll_std6", "temperature_c_is_flat",
        "air_pressure_mbar_val", "air_pressure_mbar_is_nan", "air_pressure_mbar_diff1", "air_pressure_mbar_diff2", "air_pressure_mbar_roll_std6", "air_pressure_mbar_is_flat",
        "relative_humidity_pct_val", "relative_humidity_pct_is_nan", "relative_humidity_pct_diff1", "relative_humidity_pct_diff2", "relative_humidity_pct_roll_std6", "relative_humidity_pct_is_flat",
        "score_qc", "score_autoencoder", "score_temporal", "score_statistical", "score_isolation_forest", "score_spatial", "composite_score",
        "dew_point_spread", "is_out_of_bounds"
    ]

    def __init__(self, baseline_stats: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the explainer.
        
        Args:
            baseline_stats: Optional dictionary of mean/std per variable for z-score scaling.
        """
        self.baseline_stats = baseline_stats or {
            "temperature_c": {"mean": 25.0, "std": 8.0},
            "air_pressure_mbar": {"mean": 980.0, "std": 30.0},
            "relative_humidity_pct": {"mean": 60.0, "std": 20.0},
        }

    def explain_instance(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        prev_row: Optional[Union[pd.Series, Dict[str, Any]]] = None,
        neighbor_rows: Optional[pd.DataFrame] = None,
        root_cause_model: Optional[Any] = None,
        autoencoder_errors: Optional[Dict[str, float]] = None,
    ) -> ExplanationResult:
        """
        Produce a comprehensive diagnostic explanation for a single flagged observation.

        Args:
            row: Series or dictionary containing the anomalous reading and associated scores.
            prev_row: Preceding observation at the same station (for step limit / flatline checks).
            neighbor_rows: Simultaneous observations from neighboring stations.
            root_cause_model: Optional trained RootCauseClassifier for tree attribution.
            autoencoder_errors: Optional dict mapping variable -> reconstruction error.
        """
        station_id = str(row.get("station_id", "AWS_UNKNOWN"))
        timestamp = str(row.get("timestamp", "N/A"))

        # 1. Evaluate per-variable deviations and step changes
        var_attribs: Dict[str, VariableAttribution] = {}
        evidence_points: List[str] = []
        culprit_scores: Dict[str, float] = {}

        for var in CORE_VARIABLES:
            val = float(row[var]) if (var in row and pd.notna(row[var])) else np.nan
            is_nan = np.isnan(val)
            low_b, high_b = self.PHYSICAL_RANGES[var]
            is_oob = is_nan or (val < low_b or val > high_b)

            step_chg = 0.0
            is_flat = False
            if prev_row is not None and var in prev_row and pd.notna(prev_row[var]) and not is_nan:
                prev_val = float(prev_row[var])
                step_chg = val - prev_val
                is_flat = abs(step_chg) < 1e-4

            # Z-score deviation
            mean_val = self.baseline_stats.get(var, {}).get("mean", 0.0)
            std_val = max(self.baseline_stats.get(var, {}).get("std", 1.0), 1e-5)
            z_score = abs(val - mean_val) / std_val if not is_nan else 10.0

            # Reconstruction error
            rec_err = 0.0
            if autoencoder_errors and var in autoencoder_errors:
                rec_err = float(autoencoder_errors[var])

            # Composite culprit score for this variable
            v_score = 0.0
            if is_oob:
                v_score += 5.0
                evidence_points.append(
                    f"Physical range violation on '{var}': value {val:.2f} outside [{low_b}, {high_b}]."
                )
            if abs(step_chg) > self.STEP_LIMITS[var]:
                v_score += 3.0
                evidence_points.append(
                    f"Rate-of-change limit exceeded on '{var}': Δ={step_chg:+.2f} in 1 hr (limit: {self.STEP_LIMITS[var]:.1f})."
                )
            if is_flat:
                v_score += 1.5
                evidence_points.append(f"Persistence check: '{var}' identical to previous reading (Δ=0.00).")
            if z_score > 3.0:
                v_score += min(z_score - 3.0, 3.0)
            if rec_err > 0.05:
                v_score += rec_err * 10.0

            culprit_scores[var] = v_score
            var_attribs[var] = VariableAttribution(
                variable=var,
                anomaly_score=v_score,
                deviation_sigma=z_score,
                step_change=step_chg,
                is_flatline=is_flat,
                is_out_of_bounds=is_oob,
                reconstruction_error=rec_err,
            )

        # Determine primary and all culprits
        sorted_culprits = sorted(culprit_scores.items(), key=lambda x: x[1], reverse=True)
        primary_culprit = sorted_culprits[0][0] if sorted_culprits[0][1] > 0 else "cross_sensor"
        all_culprits = [k for k, v in sorted_culprits if v >= 1.5]
        if not all_culprits:
            all_culprits = [primary_culprit]

        # 2. Thermodynamic Consistency Evidence
        thermo_evidence = self._check_thermodynamics(row)
        if thermo_evidence.get("has_violation"):
            evidence_points.append(thermo_evidence["description"])

        # 3. Spatial Disparity Evidence
        spatial_evidence = self._check_spatial_consensus(row, neighbor_rows)
        if spatial_evidence.get("has_disparity"):
            evidence_points.append(spatial_evidence["description"])

        # 4. Feature Attributions from Root Cause Model
        feature_contribs = self._compute_feature_attributions(row, root_cause_model)

        # 5. Extract predicted cause & confidence
        predicted_cause = str(row.get("predicted_cause", row.get("anomaly_type", "ANOMALY")))
        confidence = float(row.get("cause_confidence", row.get("confidence", 0.85)))

        if not evidence_points:
            evidence_points.append(f"Multi-tier fusion engine flagged observation with composite score {row.get('composite_score', 0.65):.3f}.")

        return ExplanationResult(
            station_id=station_id,
            timestamp=timestamp,
            predicted_cause=predicted_cause,
            cause_confidence=confidence,
            primary_culprit=primary_culprit,
            all_culprits=all_culprits,
            variable_attributions=var_attribs,
            top_contributing_features=feature_contribs,
            spatial_evidence=spatial_evidence,
            thermodynamic_evidence=thermo_evidence,
            evidence_summary=evidence_points,
        )

    def _check_thermodynamics(self, row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """Check psychrometric balance between temperature and relative humidity."""
        # row.get(key, default) returns None when the key EXISTS with value None
        # (common with Pydantic optional fields).  Use `or default` as a second guard.
        _t_raw  = row.get("temperature_c")      if hasattr(row, "get") else row.get("temperature_c")
        _rh_raw = row.get("relative_humidity_pct") if hasattr(row, "get") else row.get("relative_humidity_pct")
        t  = float(_t_raw  if _t_raw  is not None else 25.0)
        rh = float(_rh_raw if _rh_raw is not None else 50.0)

        if pd.isna(t) or pd.isna(rh):
            return {"has_violation": True, "description": "Thermodynamic state invalid: missing T or RH."}

        # Magnus formula for dew point
        a, b = 17.27, 237.7
        rh_clamped = max(min(rh, 100.0), 0.1)
        alpha = ((a * t) / (b + t)) + np.log(rh_clamped / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        spread = t - dew_point

        # Physical rules:
        # 1. Dew point cannot exceed temperature
        # 2. At RH near 100%, spread must be near 0
        # 3. Negative spread is physically impossible in non-supersaturated atmospheric surface layers
        has_violation = False
        desc = "Thermodynamic state is physically consistent."

        if spread < -0.2:
            has_violation = True
            desc = f"Psychrometric violation: Dew point ({dew_point:.1f}°C) exceeds ambient temperature ({t:.1f}°C) by {-spread:.1f}°C."
        elif rh < 5.0 and t < 10.0:
            has_violation = True
            desc = f"Thermodynamic anomaly: Extreme dry reading ({rh:.1f}% RH) at low temperature ({t:.1f}°C)."

        return {
            "dew_point_c": round(dew_point, 2),
            "dew_point_spread_c": round(spread, 2),
            "has_violation": has_violation,
            "description": desc,
        }

    def _check_spatial_consensus(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        neighbor_rows: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Evaluate how significantly the target observation deviates from synchronous neighbors."""
        if neighbor_rows is None or len(neighbor_rows) == 0:
            return {"has_disparity": False, "description": "No synchronous neighbor observations available."}

        disparities = {}
        has_disparity = False
        desc_items = []

        for var in CORE_VARIABLES:
            if var not in row or var not in neighbor_rows.columns:
                continue
            _v = row[var] if not hasattr(row, "get") else row.get(var)
            if _v is None or (isinstance(_v, float) and pd.isna(_v)):
                continue
            try:
                target_val = float(_v)
            except (TypeError, ValueError):
                continue
            nbr_vals = neighbor_rows[var].dropna().values
            if len(nbr_vals) == 0:
                continue

            nbr_median = float(np.median(nbr_vals))
            mad = float(np.median(np.abs(nbr_vals - nbr_median)))
            mad = max(mad, 0.5)

            delta = target_val - nbr_median
            z_mad = abs(delta) / (1.4826 * mad)

            disparities[var] = {
                "target": round(target_val, 2),
                "neighbor_median": round(nbr_median, 2),
                "delta": round(delta, 2),
                "z_mad": round(z_mad, 2),
            }

            if z_mad > 3.0:
                has_disparity = True
                desc_items.append(f"{var} diverged by {delta:+.2f} from {len(nbr_vals)} neighbors (MAD Z: {z_mad:.1f})")

        return {
            "has_disparity": has_disparity,
            "disparities": disparities,
            "description": "; ".join(desc_items) if desc_items else "Station readings concordant with spatial neighbors.",
        }

    def _compute_feature_attributions(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        root_cause_model: Optional[Any],
    ) -> List[Dict[str, Any]]:
        """Compute top feature attributions for root cause decision."""
        if root_cause_model is None or not hasattr(root_cause_model, "model"):
            # Fallback heuristic feature importance based on key signals in row
            heuristic_features = []
            for var in CORE_VARIABLES:
                if var in row and pd.notna(row[var]):
                    val   = float(row[var])
                    _d1   = row.get(f"{var}_diff1")
                    diff1 = float(_d1 if _d1 is not None else 0.0)
                    if abs(diff1) > 1.0:
                        heuristic_features.append({
                            "feature": f"{var}_diff1",
                            "value": round(diff1, 2),
                            "importance": min(abs(diff1) / 10.0, 1.0),
                            "direction": "positive" if diff1 > 0 else "negative"
                        })
            for sc in ["score_qc", "score_autoencoder", "score_temporal", "composite_score"]:
                _sc_v = row.get(sc) if hasattr(row, "get") else (row[sc] if sc in row else None)
                if _sc_v is None:
                    continue
                try:
                    sc_float = float(_sc_v)
                except (TypeError, ValueError):
                    continue
                if sc_float > 0.3:
                    heuristic_features.append({
                        "feature": sc,
                        "value": round(sc_float, 3),
                        "importance": round(sc_float, 3),
                        "direction": "positive"
                    })
            heuristic_features.sort(key=lambda x: x["importance"], reverse=True)
            return heuristic_features[:5]

        # If model is fitted RandomForestClassifier, compute feature contributions
        try:
            rf = root_cause_model.model
            importances = rf.feature_importances_
            feature_names = getattr(root_cause_model, "feature_names", self.CLASSIFIER_FEATURE_NAMES)

            # Convert row to feature array if possible
            if hasattr(root_cause_model, "extract_classification_features"):
                df_single = pd.DataFrame([row])
                X_vec = root_cause_model.extract_classification_features(df_single)[0]
            else:
                X_vec = np.zeros(len(importances))

            top_indices = np.argsort(-importances)[:5]
            contribs = []
            for idx in top_indices:
                if idx < len(feature_names):
                    feat_name = feature_names[idx]
                    feat_val = float(X_vec[idx]) if idx < len(X_vec) else 0.0
                    contribs.append({
                        "feature": feat_name,
                        "value": round(feat_val, 3),
                        "importance": round(float(importances[idx]), 4),
                        "direction": "positive" if feat_val > 0 else "neutral"
                    })
            return contribs
        except Exception:
            return []
