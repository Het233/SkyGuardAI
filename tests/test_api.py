"""
================================================================================
SkyGuard AI — Test Suite for Real-Time REST API (Phase 11)
================================================================================
"""

import unittest
from fastapi.testclient import TestClient

from api.server import app, manager


class TestRestAPI(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        manager.load_models()

    def test_health_check(self):
        """Verify system liveness endpoint returns ONLINE."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("device", data)

    def test_ingest_single_nominal_observation(self):
        """Verify nominal sensor reading passes through without anomaly flags."""
        payload = {
            "station_id": "IMD_TEST_01",
            "timestamp": "2024-01-01T12:00:00",
            "temperature_c": 28.5,
            "air_pressure_mbar": 1008.2,
            "relative_humidity_pct": 58.0,
            "latitude": 12.9716,
            "longitude": 77.5946,
            "elevation_m": 920.0,
        }

        response = self.client.post("/api/v1/ingest/single", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["station_id"], "IMD_TEST_01")
        self.assertFalse(data["is_anomaly"])
        self.assertIn(data["severity"], ["NONE", "LOW", "NORMAL", "SUSPICIOUS"])
        # Original value preserved in imputed
        self.assertEqual(data["imputed_temperature_c"], 28.5)
        self.assertEqual(data["imputation_flag"], "RAW_PASSTHROUGH")

    def test_ingest_single_critical_anomaly(self):
        """Verify that gross physical violation triggers CRITICAL severity and non-destructive correction."""
        payload = {
            "station_id": "IMD_TEST_02",
            "timestamp": "2024-01-01T12:00:00",
            "temperature_c": 999.0,  # Gross impossible temperature
            "air_pressure_mbar": 1005.0,
            "relative_humidity_pct": 50.0,
        }

        response = self.client.post("/api/v1/ingest/single", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["is_anomaly"])
        self.assertIn(data["severity"], ["HIGH", "CRITICAL", "ANOMALY"])
        self.assertGreater(data["composite_score"], 0.70)
        # Imputed temperature must be corrected and physically bounded (<= 55°C)
        self.assertLessEqual(data["imputed_temperature_c"], 55.0)
        self.assertNotEqual(data["imputed_temperature_c"], 999.0)
        # Diagnostic card must be generated
        self.assertIsNotNone(data["diagnostic_card"])
        self.assertIn("alert_header", data["diagnostic_card"])

    def test_ingest_batch_observations(self):
        """Verify batch ingest processing across multiple stations."""
        payload = {
            "observations": [
                {
                    "station_id": "IMD_BATCH_01",
                    "timestamp": "2024-01-01T12:00:00",
                    "temperature_c": 26.0,
                    "air_pressure_mbar": 1012.0,
                    "relative_humidity_pct": 65.0,
                },
                {
                    "station_id": "IMD_BATCH_02",
                    "timestamp": "2024-01-01T12:00:00",
                    "temperature_c": -100.0,  # Impossible reading
                    "air_pressure_mbar": 1012.0,
                    "relative_humidity_pct": 65.0,
                },
            ]
        }

        response = self.client.post("/api/v1/ingest/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["total_processed"], 2)
        self.assertGreaterEqual(data["anomalies_detected"], 1)
        self.assertEqual(len(data["results"]), 2)

    def test_station_health_endpoints(self):
        """Verify station health querying returns 0-100 score and status."""
        # Seed an observation to ensure station buffer is populated
        seed_payload = {
            "station_id": "IMD_SEED_01",
            "timestamp": "2024-01-01T12:00:00",
            "temperature_c": 27.0,
            "air_pressure_mbar": 1010.0,
            "relative_humidity_pct": 50.0,
        }
        self.client.post("/api/v1/ingest/single", json=seed_payload)

        response = self.client.get("/api/v1/stations/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        first_st = data[0]["station_id"]
        single_rep = self.client.get(f"/api/v1/stations/{first_st}/health")
        self.assertEqual(single_rep.status_code, 200)
        rep_data = single_rep.json()
        self.assertIn("overall_score", rep_data)
        self.assertIn("status", rep_data)
        self.assertIn("sensors", rep_data)

    def test_active_alerts_endpoint(self):
        """Verify active alerts endpoint returns alert queue."""
        response = self.client.get("/api/v1/alerts/active")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("active_alert_count", data)
        self.assertIn("alerts", data)


if __name__ == "__main__":
    unittest.main()
