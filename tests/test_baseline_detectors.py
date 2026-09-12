"""
================================================================================
SkyGuard AI — Test Suite for Baseline QC & Statistical Detectors
================================================================================
"""

import unittest
import numpy as np
import pandas as pd

from baseline.quality_control import DeterministicQCValidator
from baseline.statistical import (
    ZScoreDetector,
    IQRDetector,
    EWMADetector,
    CompositeBaselineDetector,
)
from evaluation.metrics import compute_binary_metrics, compute_type_breakdown


def make_clean_station_data(n: int = 100, station_id: str = "TEST_STATION") -> pd.DataFrame:
    """Create clean synthetic sine-wave station data."""
    t = pd.date_range("2024-01-01", periods=n, freq="1h")
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 24.0)
    press = 1010.0 + 2.0 * np.cos(2 * np.pi * np.arange(n) / 24.0)
    rh = 60.0 - 20.0 * np.sin(2 * np.pi * np.arange(n) / 24.0)

    return pd.DataFrame({
        "station_id": station_id,
        "timestamp": t,
        "temperature_c": temp,
        "air_pressure_mbar": press,
        "relative_humidity_pct": rh,
    })


class TestDeterministicQC(unittest.TestCase):
    """Verify physical range, step change, and persistence checks."""

    def test_out_of_bounds_detection(self):
        df = make_clean_station_data(50)
        # Inject out of bounds
        df.loc[10, "temperature_c"] = 85.0       # > 60
        df.loc[20, "relative_humidity_pct"] = 125.0 # > 100
        df.loc[30, "air_pressure_mbar"] = 150.0  # < 300

        validator = DeterministicQCValidator()
        flags, details = validator.validate(df)

        self.assertTrue(flags.iloc[10])
        self.assertTrue(flags.iloc[20])
        self.assertTrue(flags.iloc[30])
        self.assertFalse(flags.iloc[5])

    def test_rate_of_change_detection(self):
        df = make_clean_station_data(50)
        # Sudden 25 °C jump between step 14 and 15
        df.loc[15, "temperature_c"] = df.loc[14, "temperature_c"] + 25.0

        validator = DeterministicQCValidator()
        flags, details = validator.validate(df)

        self.assertTrue(flags.iloc[15])
        self.assertTrue(details.loc[15, "temperature_c_rate_of_change_exceeded"])

    def test_persistence_frozen_sensor(self):
        df = make_clean_station_data(50)
        # Hold temperature constant for 8 steps (indices 20 to 27)
        flat_val = float(df.loc[20, "temperature_c"])
        for i in range(20, 28):
            df.loc[i, "temperature_c"] = flat_val

        validator = DeterministicQCValidator(persistence_steps=6)
        flags, details = validator.validate(df)

        # Frozen points should be flagged
        self.assertTrue(flags.iloc[25])
        self.assertTrue(details.loc[25, "temperature_c_frozen"])


class TestStatisticalDetectors(unittest.TestCase):
    """Verify Z-Score, IQR, EWMA, and Composite detectors."""

    def test_z_score_detector(self):
        df = make_clean_station_data(100)
        # Inject spike
        df.loc[50, "temperature_c"] += 25.0

        detector = ZScoreDetector(window=24, threshold=3.5)
        flags, scores = detector.detect(df)

        self.assertEqual(flags.iloc[50], 1)
        self.assertGreater(scores.iloc[50], 0.7)
        # Verify clean point is not flagged
        self.assertEqual(flags.iloc[20], 0)

    def test_iqr_detector(self):
        df = make_clean_station_data(100)
        df.loc[60, "air_pressure_mbar"] -= 40.0

        detector = IQRDetector(window=48, multiplier=1.8)
        flags, scores = detector.detect(df)

        self.assertEqual(flags.iloc[60], 1)
        self.assertGreater(scores.iloc[60], 0.5)

    def test_ewma_detector(self):
        df = make_clean_station_data(100)
        # Significant surge relative to RH variance (sigma ~ 14.1)
        df.loc[70, "relative_humidity_pct"] += 60.0

        detector = EWMADetector(span=24, sigma_threshold=3.5)
        flags, scores = detector.detect(df)

        self.assertEqual(flags.iloc[70], 1)

    def test_composite_baseline_detector(self):
        df = make_clean_station_data(100)
        # Out of bounds
        df.loc[30, "temperature_c"] = 90.0
        # High spike
        df.loc[60, "air_pressure_mbar"] -= 50.0

        detector = CompositeBaselineDetector()
        flags, scores, components = detector.detect(df)

        self.assertEqual(flags.iloc[30], 1)
        self.assertEqual(flags.iloc[60], 1)
        self.assertEqual(scores.iloc[30], 1.0)
        self.assertIn("qc_flag", components.columns)
        self.assertIn("z_flag", components.columns)


class TestEvaluationMetrics(unittest.TestCase):
    """Verify evaluation metric computations."""

    def test_binary_metrics_calculation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.4, 0.6, 0.1])

        metrics = compute_binary_metrics(y_true, y_pred, y_score, num_stations=1)

        # TP=2, FP=1, TN=4, FN=1
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 4)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertAlmostEqual(metrics["precision"], 2 / 3, places=3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3, places=3)
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertIsNotNone(metrics["pr_auc"])

    def test_type_breakdown(self):
        df = pd.DataFrame({
            "anomaly_type": ["NORMAL", "SPIKE", "SPIKE", "SENSOR_DRIFT", "NORMAL"],
            "pred": [0, 1, 1, 0, 0],
        })
        breakdown = compute_type_breakdown(df, y_pred_col="pred", truth_type_col="anomaly_type")

        self.assertIn("SPIKE", breakdown)
        self.assertEqual(breakdown["SPIKE"]["total_instances"], 2)
        self.assertEqual(breakdown["SPIKE"]["detected"], 2)
        self.assertEqual(breakdown["SPIKE"]["recall"], 1.0)

        self.assertIn("SENSOR_DRIFT", breakdown)
        self.assertEqual(breakdown["SENSOR_DRIFT"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
