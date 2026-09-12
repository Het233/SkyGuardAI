"""
================================================================================
SkyGuard AI — Test Suite for Phases 8, 9 & 10
(Explainable AI, Sensor Health Scoring, and Value Imputation)
================================================================================
"""

import unittest
import numpy as np
import pandas as pd

from explainability.explainer import AnomalyExplainer, ExplanationResult
from explainability.diagnostic_card import DiagnosticCardGenerator
from health.health_score import SensorHealthTracker, HealthStatus
from imputation.correction import MeteorologicalImputer, ImputationFlag


class TestExplainableAI(unittest.TestCase):
    """Test suite for Explainable AI (XAI) feature attribution and alert cards."""

    def setUp(self):
        self.explainer = AnomalyExplainer()

    def test_temperature_spike_attribution(self):
        """Verify that a sudden temperature step jump attributes temperature_c as the primary culprit."""
        prev_row = {
            "station_id": "AWS_001",
            "timestamp": "2024-01-01 12:00:00",
            "temperature_c": 25.0,
            "air_pressure_mbar": 1005.0,
            "relative_humidity_pct": 55.0,
        }
        curr_row = {
            "station_id": "AWS_001",
            "timestamp": "2024-01-01 13:00:00",
            "temperature_c": 42.0,  # +17°C jump in 1 hr (exceeds 12°C limit)
            "air_pressure_mbar": 1005.0,
            "relative_humidity_pct": 55.0,
            "anomaly_type": "SPIKE",
            "confidence": 0.94,
        }

        result = self.explainer.explain_instance(curr_row, prev_row=prev_row)

        self.assertIsInstance(result, ExplanationResult)
        self.assertEqual(result.primary_culprit, "temperature_c")
        self.assertTrue(any("Rate-of-change limit exceeded" in ev for ev in result.evidence_summary))
        self.assertTrue(result.variable_attributions["temperature_c"].step_change > 12.0)

    def test_thermodynamic_violation_detection(self):
        """Verify that an impossible thermodynamic state (dew point > temperature) is flagged."""
        # High RH with extreme temperature anomaly
        row = {
            "station_id": "AWS_002",
            "timestamp": "2024-01-01 14:00:00",
            "temperature_c": -10.0,
            "air_pressure_mbar": 1000.0,
            "relative_humidity_pct": 105.0,  # Invalid RH
            "anomaly_type": "PHYSICALLY_IMPOSSIBLE",
        }

        result = self.explainer.explain_instance(row)
        self.assertTrue(result.variable_attributions["relative_humidity_pct"].is_out_of_bounds)
        self.assertTrue(len(result.evidence_summary) > 0)

    def test_diagnostic_card_generation(self):
        """Verify alert card formatting and prescriptive maintenance recommendations."""
        row = {
            "station_id": "AWS_003",
            "timestamp": "2024-01-01 15:00:00",
            "temperature_c": 26.0,
            "air_pressure_mbar": 995.0,
            "relative_humidity_pct": 40.0,
            "predicted_cause": "FROZEN_SENSOR",
            "cause_confidence": 0.98,
        }
        explanation = self.explainer.explain_instance(row)
        card = DiagnosticCardGenerator.generate_card(explanation, severity="CRITICAL", composite_score=0.92)

        self.assertIn("alert_header", card)
        self.assertEqual(card["alert_header"]["predicted_fault"], "FROZEN_SENSOR")
        self.assertEqual(card["alert_header"]["urgency"], "CRITICAL")
        self.assertIn("maintenance_prescription", card)
        self.assertIn("action", card["maintenance_prescription"])

        md = DiagnosticCardGenerator.format_markdown(card)
        self.assertIn("Diagnostic Alert — Station AWS_003", md)
        self.assertIn("FROZEN_SENSOR", md)


class TestSensorHealthScoring(unittest.TestCase):
    """Test suite for sensor health degradation and predictive RUL."""

    def setUp(self):
        self.tracker = SensorHealthTracker(rolling_window_hours=72)

    def test_clean_station_health(self):
        """Verify pristine data yields EXCELLENT health (>90.0)."""
        dates = pd.date_range("2024-01-01", periods=72, freq="1h")
        df_clean = pd.DataFrame({
            "station_id": ["AWS_101"] * 72,
            "timestamp": dates,
            "temperature_c": 25.0 + 5.0 * np.sin(np.linspace(0, 6 * np.pi, 72)),
            "air_pressure_mbar": 1010.0 + 2.0 * np.cos(np.linspace(0, 6 * np.pi, 72)),
            "relative_humidity_pct": 60.0 + 10.0 * np.cos(np.linspace(0, 6 * np.pi, 72)),
            "is_anomaly": [0] * 72,
        })

        report = self.tracker.compute_station_health(df_clean)
        self.assertGreaterEqual(report.overall_score, 90.0)
        self.assertEqual(report.status, HealthStatus.EXCELLENT)
        self.assertEqual(report.urgency, "ROUTINE")

    def test_degrading_station_penalties(self):
        """Verify repeated anomalies significantly depress health scores to DEGRADING or CRITICAL."""
        dates = pd.date_range("2024-01-01", periods=72, freq="1h")
        # 30% anomalous readings
        anom_flags = np.array([1 if i % 3 == 0 else 0 for i in range(72)])
        df_degraded = pd.DataFrame({
            "station_id": ["AWS_102"] * 72,
            "timestamp": dates,
            "temperature_c": [25.0] * 72,  # Complete flatline
            "air_pressure_mbar": 1010.0 + np.random.normal(0, 0.2, 72),
            "relative_humidity_pct": 60.0 + np.random.normal(0, 1.0, 72),
            "is_anomaly": anom_flags,
        })

        report = self.tracker.compute_station_health(df_degraded)
        self.assertLess(report.overall_score, 75.0)
        self.assertIn(report.status, [HealthStatus.DEGRADING, HealthStatus.CRITICAL])
        self.assertTrue(len(report.work_orders) > 0)


class TestValueImputation(unittest.TestCase):
    """Test suite for non-destructive spatial-temporal value imputation."""

    def setUp(self):
        self.imputer = MeteorologicalImputer(k_neighbors=2)

    def test_non_destructive_preservation(self):
        """Verify that original raw columns are NEVER overwritten or modified."""
        df = pd.DataFrame({
            "station_id": ["AWS_A", "AWS_A"],
            "timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
            "temperature_c": [25.0, 999.0],  # 999.0 is gross physical anomaly
            "air_pressure_mbar": [1000.0, 1000.0],
            "relative_humidity_pct": [50.0, 50.0],
            "is_anomaly": [False, True],
        })

        df_imputed = self.imputer.impute_dataframe(df)

        # Raw column must remain completely identical
        self.assertEqual(df_imputed["temperature_c"].iloc[1], 999.0)
        # Imputed column must contain corrected realistic value
        self.assertNotEqual(df_imputed["imputed_temperature_c"].iloc[1], 999.0)
        self.assertLess(df_imputed["imputed_temperature_c"].iloc[1], 55.0)
        self.assertIn("imputation_flag", df_imputed.columns)
        self.assertIn("imputation_confidence", df_imputed.columns)

    def test_spatial_lapse_elevation_correction(self):
        """Verify that spatial neighbor imputation adjusts for elevation lapse rate."""
        # Station A at sea level (0m), Station B at hill station (1000m)
        # Expected temp at Station B is lower by ~6.5°C
        df = pd.DataFrame({
            "station_id": ["AWS_A", "AWS_B"],
            "latitude": [12.0, 12.1],
            "longitude": [77.0, 77.1],
            "elevation_m": [0.0, 1000.0],
            "timestamp": ["2024-01-01 12:00:00", "2024-01-01 12:00:00"],
            "temperature_c": [30.0, np.nan],  # Station B missing reading
            "air_pressure_mbar": [1013.0, np.nan],
            "relative_humidity_pct": [60.0, np.nan],
            "is_anomaly": [False, True],
        })

        imputer = MeteorologicalImputer(k_neighbors=1, max_neighbor_distance_km=200.0)
        df_imputed = imputer.impute_dataframe(df)

        # Target B elevation = 1000m, Neighbor A elevation = 0m. delta_elev = +1000m
        # Lapse rate = -0.0065 * 1000 = -6.5°C
        # Expected T at B = 30.0 - 6.5 = 23.5°C
        b_imputed_temp = df_imputed.loc[df_imputed["station_id"] == "AWS_B", "imputed_temperature_c"].iloc[0]
        self.assertAlmostEqual(b_imputed_temp, 23.5, delta=0.5)

    def test_psychrometric_balance_enforcement(self):
        """Verify that imputed relative humidity and dew point respect physical laws."""
        vals = {"temperature_c": 15.0, "air_pressure_mbar": 1012.0, "relative_humidity_pct": 120.0}
        corrected = MeteorologicalImputer._enforce_psychrometric_equilibrium(vals)

        self.assertLessEqual(corrected["relative_humidity_pct"], 100.0)
        self.assertGreaterEqual(corrected["relative_humidity_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
