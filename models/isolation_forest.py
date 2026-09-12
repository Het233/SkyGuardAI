"""
================================================================================
SkyGuard AI — Weather Isolation Forest Detector
================================================================================
Unsupervised tree-based anomaly detector trained on clean AWS feature representations.
Isolates multi-dimensional anomalies based on tree path depths.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class WeatherIsolationForest:
    """
    Scikit-learn Isolation Forest wrapper customized for meteorological anomaly detection.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_samples: Union[int, float] = "auto",
        contamination: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.threshold: float = 0.5
        self.score_min: float = -1.0
        self.score_max: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "WeatherIsolationForest":
        """
        Fit Isolation Forest on clean training features.
        """
        self.model.fit(X)
        self.is_fitted = True

        # Calibrate score scaling on training distribution
        raw_scores = self.model.score_samples(X)
        self.score_min = float(np.min(raw_scores))
        self.score_max = float(np.max(raw_scores))

        # Calibrate decision threshold at the specified contamination percentile
        self.threshold = float(np.percentile(self._normalize_scores(raw_scores), 100.0 * (1.0 - self.contamination)))

        return self

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Convert raw decision scores (where negative = anomalous) to [0.0, 1.0]
        where 1.0 represents extreme anomaly.
        """
        # Invert: lowest raw score becomes highest anomaly score
        spread = max(1e-6, self.score_max - self.score_min)
        norm = (self.score_max - raw_scores) / spread
        return np.clip(norm, 0.0, 1.0)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous anomaly scores in [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        raw_scores = self.model.score_samples(X)
        return self._normalize_scores(raw_scores)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict binary anomalies and continuous scores.

        Returns:
            Tuple of (binary_flag_array [0 or 1], continuous_score_array).
        """
        scores = self.score(X)
        binary_preds = (scores >= self.threshold).astype(int)
        return binary_preds, scores

    def save(self, filepath: str) -> None:
        """Serialize model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "WeatherIsolationForest":
        """Load serialized model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
