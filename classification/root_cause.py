"""
================================================================================
SkyGuard AI — Root-Cause Multiclass Fault Classifier
================================================================================
Supervised machine learning model identifying the specific operational root cause
behind flagged meteorological anomalies (Spike, Drift, Freeze, Noise, etc.).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from anomaly_injection.taxonomy import AnomalyType, CORE_VARIABLES


def attach_detector_scores(df: pd.DataFrame, models_dir: str) -> pd.DataFrame:
    """Helper to compute and attach component detector scores to a DataFrame."""
    df = df.reset_index(drop=True)
    fe_path = os.path.join(models_dir, "feature_engineer.pkl")
    iso_path = os.path.join(models_dir, "isolation_forest.pkl")
    ae_path = os.path.join(models_dir, "autoencoder.pt")
    tp_path = os.path.join(models_dir, "temporal_predictor.pkl")

    if not (os.path.exists(fe_path) and os.path.exists(iso_path) and os.path.exists(ae_path) and os.path.exists(tp_path)):
        return df

    from features.engineering import WeatherFeatureEngineer
    from models.isolation_forest import WeatherIsolationForest
    from models.autoencoder import DeepWeatherAutoencoder
    from models.temporal_model import TemporalResidualPredictor
    from baseline.quality_control import DeterministicQCValidator
    from baseline.statistical import CompositeBaselineDetector

    with open(fe_path, "rb") as f:
        engineer = pickle.load(f)
    iso_model = WeatherIsolationForest.load(iso_path)
    ae_model = DeepWeatherAutoencoder.load(ae_path)
    tp_model = TemporalResidualPredictor.load(tp_path)

    qc = DeterministicQCValidator()
    stat = CompositeBaselineDetector()

    X_scaled = engineer.transform(df)
    y_core = df[CORE_VARIABLES].values

    qc_flags, _ = qc.validate(df)
    _, stat_scores, _ = stat.detect(df)
    iso_scores = np.nan_to_num(iso_model.score(X_scaled), nan=0.5)
    ae_scores = np.nan_to_num(ae_model.score(X_scaled), nan=0.5)
    temp_scores, _ = tp_model.score(X_scaled, y_core)
    temp_scores = np.nan_to_num(temp_scores, nan=1.0)
    qc_val = np.nan_to_num(qc_flags.astype(float).values, nan=0.0)

    df_out = df.copy()
    df_out["score_qc"] = qc_val
    df_out["score_statistical"] = np.nan_to_num(stat_scores.values, nan=0.0)
    df_out["score_isolation_forest"] = iso_scores
    df_out["score_autoencoder"] = ae_scores
    df_out["score_temporal"] = temp_scores
    df_out["composite_score"] = np.maximum.reduce([qc_val, ae_scores, temp_scores])
    return df_out


class RootCauseClassifier:
    """
    Multiclass classifier attributing anomalies to precise physical fault taxonomy classes.
    """

    SUPPORTED_CLASSES = [
        AnomalyType.NORMAL.value,
        AnomalyType.SPIKE.value,
        AnomalyType.DROP.value,
        AnomalyType.FROZEN_SENSOR.value,
        AnomalyType.SENSOR_DRIFT.value,
        AnomalyType.CONSTANT_OFFSET.value,
        AnomalyType.HIGH_NOISE.value,
        AnomalyType.MISSING_DATA.value,
        AnomalyType.COMMUNICATION_CORRUPTION.value,
        AnomalyType.PHYSICALLY_IMPOSSIBLE.value,
        AnomalyType.CROSS_SENSOR_INCONSISTENCY.value,
        AnomalyType.COORDINATED_MULTIVARIATE.value,
    ]

    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        self.is_fitted: bool = False
        self.classes_: List[str] = []

    def extract_classification_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract discriminative feature signatures for fault classification.
        """
        df = df.reset_index(drop=True)
        feat_df = pd.DataFrame(index=df.index)

        # 1. Base Variables, Imputed Signals, and Missingness
        for var in CORE_VARIABLES:
            if var in df.columns:
                series = df[var]
                is_nan = series.isna().astype(float)
                med_val = series.median() if len(series.dropna()) > 0 else 0.0
                clean_series = series.ffill().bfill().fillna(med_val)

                feat_df[f"{var}_val"] = clean_series
                feat_df[f"{var}_is_nan"] = is_nan

                # Station-isolated or global differences
                if "station_id" in df.columns:
                    diff1 = df.groupby("station_id")[var].diff().fillna(0.0)
                    diff2 = diff1.diff().fillna(0.0)
                    roll_std6 = df.groupby("station_id")[var].rolling(6, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
                    is_flat = (diff1.abs() < 1e-5).astype(float)
                else:
                    diff1 = clean_series.diff().fillna(0.0)
                    diff2 = diff1.diff().fillna(0.0)
                    roll_std6 = clean_series.rolling(6, min_periods=2).std().fillna(0.0)
                    is_flat = (diff1.abs() < 1e-5).astype(float)

                feat_df[f"{var}_diff1"] = diff1
                feat_df[f"{var}_diff2"] = diff2
                feat_df[f"{var}_roll_std6"] = roll_std6
                feat_df[f"{var}_is_flat"] = is_flat
            else:
                feat_df[f"{var}_val"] = 0.0
                feat_df[f"{var}_is_nan"] = 1.0
                feat_df[f"{var}_diff1"] = 0.0
                feat_df[f"{var}_diff2"] = 0.0
                feat_df[f"{var}_roll_std6"] = 0.0
                feat_df[f"{var}_is_flat"] = 0.0

        # 3. Component Detector Scores (if available in df)
        score_cols = [
            "score_qc",
            "score_autoencoder",
            "score_temporal",
            "score_statistical",
            "score_isolation_forest",
            "score_spatial",
            "composite_score",
        ]
        for sc in score_cols:
            if sc in df.columns:
                feat_df[sc] = df[sc].fillna(0.0)
            else:
                feat_df[sc] = 0.0

        # 4. Thermodynamic psychrometric proxy
        # Cast to float64 explicitly — a 1-row DataFrame built from Python
        # scalars may have object-dtype columns that break np.log (a ufunc).
        t  = df["temperature_c"].fillna(25.0).to_numpy(dtype=np.float64)
        rh = np.clip(
            df["relative_humidity_pct"].fillna(50.0).to_numpy(dtype=np.float64),
            1.0, 100.0,
        )
        a, b = 17.27, 237.7
        alpha = ((a * t) / (b + t)) + np.log(rh / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        feat_df["dew_point_spread"] = t - dew_point

        # 5. Out-of-bounds flag
        is_oob = (
            (df["temperature_c"] < -50.0) | (df["temperature_c"] > 60.0) |
            (df["air_pressure_mbar"] < 300.0) | (df["air_pressure_mbar"] > 1100.0) |
            (df["relative_humidity_pct"] < 0.0) | (df["relative_humidity_pct"] > 100.0)
        ).fillna(False).astype(float)
        feat_df["is_out_of_bounds"] = is_oob

        return feat_df.fillna(0.0).values

    def fit(self, df: pd.DataFrame, target_col: str = "anomaly_type") -> "RootCauseClassifier":
        """
        Train the multiclass classifier on ground-truth labeled anomaly data.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        X = self.extract_classification_features(df)
        y = df[target_col].astype(str).values

        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict root cause labels and confidence probabilities.

        Returns:
            Tuple of:
                - predicted_classes: Array of predicted fault type strings.
                - confidence_scores: Array of maximum class probabilities in [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict().")

        X = self.extract_classification_features(df)
        probs = self.model.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)

        pred_classes = np.array([self.classes_[i] for i in pred_indices])
        confidences = np.max(probs, axis=1)

        return pred_classes, confidences

    def get_top_causes(self, df: pd.DataFrame, top_k: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Return top-K candidate causes with associated probabilities for each row.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before scoring.")

        X = self.extract_classification_features(df)
        probs = self.model.predict_proba(X)

        results = []
        for row_probs in probs:
            top_indices = np.argsort(-row_probs)[:top_k]
            row_candidates = [
                {"cause": self.classes_[i], "probability": round(float(row_probs[i]), 3)}
                for i in top_indices
            ]
            results.append(row_candidates)

        return results

    def save(self, filepath: str) -> None:
        """Serialize model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "RootCauseClassifier":
        """Load serialized classifier from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
