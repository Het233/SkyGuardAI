"""
================================================================================
SkyGuard AI — Test Suite for Feature Engineering
================================================================================
"""

import unittest
import numpy as np
import pandas as pd

from features.engineering import WeatherFeatureEngineer
from anomaly_injection.taxonomy import CORE_VARIABLES


class TestWeatherFeatureEngineer(unittest.TestCase):
    """Test feature engineering pipeline."""

    def setUp(self):
        records = []
        base_time = pd.Timestamp("2024-01-01 00:00:00")
        for s_idx in range(2):
            st = f"STATION_{s_idx}"
            for i in range(50):
                records.append({
                    "station_id": st,
                    "timestamp": str(base_time + pd.Timedelta(hours=i)),
                    "temperature_c": 25.0 + 5.0 * np.sin(2 * np.pi * i / 24.0),
                    "air_pressure_mbar": 1010.0 + 2.0 * np.cos(2 * np.pi * i / 24.0),
                    "relative_humidity_pct": 60.0 - 20.0 * np.sin(2 * np.pi * i / 24.0),
                })
        self.df = pd.DataFrame(records)

    def test_feature_extraction(self):
        engineer = WeatherFeatureEngineer()
        X_df = engineer.extract_features(self.df)

        self.assertFalse(X_df.isna().any().any(), "Features should not contain any NaNs.")
        self.assertGreater(len(X_df.columns), 20, "Should generate comprehensive feature set.")

        # Verify core variable lags exist
        for var in CORE_VARIABLES:
            self.assertIn(f"{var}_lag_1", X_df.columns)
            self.assertIn(f"{var}_roll_mean_24", X_df.columns)

        # Verify thermodynamic features exist
        self.assertIn("dew_point_spread", X_df.columns)
        self.assertIn("vapor_pressure_proxy", X_df.columns)
        self.assertIn("sin_hour", X_df.columns)

    def test_fit_transform(self):
        engineer = WeatherFeatureEngineer()
        X_scaled = engineer.fit_transform(self.df)

        self.assertIsInstance(X_scaled, np.ndarray)
        self.assertEqual(X_scaled.shape[0], len(self.df))
        self.assertEqual(X_scaled.shape[1], len(engineer.feature_names))

        # Check approximately zero-mean after standard scaling
        col_means = np.mean(X_scaled, axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
