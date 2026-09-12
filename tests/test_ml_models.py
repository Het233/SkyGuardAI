"""
================================================================================
SkyGuard AI — Test Suite for Machine Learning & Deep Learning Detectors
================================================================================
"""

import os
import unittest
import numpy as np
import pandas as pd

from features.engineering import WeatherFeatureEngineer
from models.isolation_forest import WeatherIsolationForest
from models.autoencoder import DeepWeatherAutoencoder
from models.temporal_model import TemporalResidualPredictor
from anomaly_injection.taxonomy import CORE_VARIABLES


def make_clean_features(n: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to generate scaled features and ground-truth target values."""
    records = []
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    for i in range(n):
        records.append({
            "station_id": "TEST_AWS",
            "timestamp": str(base_time + pd.Timedelta(hours=i)),
            "temperature_c": 25.0 + 6.0 * np.sin(2 * np.pi * i / 24.0),
            "air_pressure_mbar": 1010.0 + 3.0 * np.cos(2 * np.pi * i / 24.0),
            "relative_humidity_pct": 65.0 - 25.0 * np.sin(2 * np.pi * i / 24.0),
        })
    df = pd.DataFrame(records)
    engineer = WeatherFeatureEngineer()
    X = engineer.fit_transform(df)
    y = df[CORE_VARIABLES].values
    return X, y


class TestMLModels(unittest.TestCase):
    """Verify Isolation Forest, Autoencoder, and Temporal Residual models."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_clean_features(150)

    def test_isolation_forest(self):
        model = WeatherIsolationForest(n_estimators=30, random_state=42)
        model.fit(self.X)

        scores = model.score(self.X)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))

        preds, _ = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_autoencoder(self):
        # Train quick autoencoder on CPU/GPU
        model = DeepWeatherAutoencoder(epochs=3, batch_size=32)
        model.fit(self.X)

        scores = model.score(self.X)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))

        preds, _ = model.predict(self.X)
        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_temporal_residual_predictor(self):
        model = TemporalResidualPredictor(alpha=1.0)
        model.fit(self.X, self.y)

        scores, var_z = model.score(self.X, self.y)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))
        self.assertEqual(var_z.shape, self.y.shape)

        preds, _ = model.predict(self.X, self.y)
        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
