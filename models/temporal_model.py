"""
================================================================================
SkyGuard AI — Temporal Residual Predictor
================================================================================
Predicts expected one-step-ahead sensor trajectories from historical lags and diurnal
cycles. Flags anomalies by standardizing the forecasting residual magnitude:
    residual = |x_t - x_pred_t|
Particularly effective at capturing gradual sensor drift and calibration steps.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from anomaly_injection.taxonomy import CORE_VARIABLES


class TemporalResidualPredictor:
    """
    Multi-target temporal prediction model predicting (T, P, RH) from recent history.
    """

    def __init__(self, alpha: float = 1.0, threshold_sigma: float = 3.5):
        self.alpha = alpha
        self.threshold_sigma = threshold_sigma

        # Multi-output Ridge regressor
        self.model = Ridge(alpha=self.alpha)
        self.residual_means: Dict[str, float] = {}
        self.residual_stds: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalResidualPredictor":
        """
        Fit temporal predictor.
        X: Feature matrix containing lags, rolling moments, and cyclical embeddings.
        y: Ground truth core sensor variables [T, P, RH].
        """
        self.model.fit(X, y)
        self.is_fitted = True

        # Compute residuals on training set to establish baseline residual standard deviation
        preds = self.model.predict(X)
        raw_residuals = np.abs(y - preds)

        for idx, var in enumerate(CORE_VARIABLES):
            var_res = raw_residuals[:, idx]
            self.residual_means[var] = float(np.mean(var_res))
            self.residual_stds[var] = float(max(0.05, np.std(var_res)))

        return self

    def predict_values(self, X: np.ndarray) -> np.ndarray:
        """Forecast expected sensor values."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute standardized prediction residual anomaly scores.

        Args:
            X: Feature matrix.
            y: Actual observed sensor values [T, P, RH].

        Returns:
            Tuple of (composite_score [0, 1], per_variable_z_scores).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        preds = self.predict_values(X)
        abs_residuals = np.abs(y - preds)

        var_z_scores = np.zeros_like(abs_residuals)
        for idx, var in enumerate(CORE_VARIABLES):
            std = self.residual_stds.get(var, 1.0)
            var_z_scores[:, idx] = abs_residuals[:, idx] / std

        # Max standardized residual across variables
        max_z = np.max(var_z_scores, axis=1)
        # Normalize: threshold_sigma corresponds to ~0.67
        norm_score = np.clip(max_z / (self.threshold_sigma * 1.5), 0.0, 1.0)

        return norm_score, var_z_scores

    def predict(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict binary anomalies and continuous scores.
        """
        scores, var_z = self.score(X, y)
        threshold_norm = self.threshold_sigma / (self.threshold_sigma * 1.5)
        binary_preds = (scores >= threshold_norm).astype(int)
        return binary_preds, scores

    def save(self, filepath: str) -> None:
        """Serialize model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "TemporalResidualPredictor":
        """Load serialized model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
