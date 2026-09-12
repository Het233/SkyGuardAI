"""
================================================================================
SkyGuard AI — Test Suite for Event-vs-Fault Reasoning & Root-Cause Classifier
================================================================================
"""

import unittest
import numpy as np
import pandas as pd

from reasoning.event_vs_fault import EventVsFaultClassifier
from classification.root_cause import RootCauseClassifier
from models.spatial_model import SpatialConsensusDetector


class TestEventVsFaultClassifier(unittest.TestCase):
    """Verify distinguishing genuine weather events from sensor faults."""

    def setUp(self):
        self.spatial = SpatialConsensusDetector(k_neighbors=2)
        self.classifier = EventVsFaultClassifier(spatial_detector=self.spatial)

    def test_physically_impossible_fault(self):
        # Impossible reading (e.g. 150 °C)
        row = pd.Series({
            "station_id": "STN_A",
            "timestamp": "2024-01-01T12:00",
            "temperature_c": 150.0,
            "air_pressure_mbar": 1010.0,
            "relative_humidity_pct": 50.0,
        })
        res = self.classifier.evaluate_observation(row, df_history=pd.DataFrame([row]))

        self.assertEqual(res["classification"], "HARDWARE_SENSOR_FAULT")
        self.assertGreaterEqual(res["p_sensor_fault"], 0.95)

    def test_regional_heatwave_genuine_event(self):
        # Target station warms to 42°C, and neighbors also warm to 41°C and 43°C with low humidity
        target_row = pd.Series({
            "station_id": "STN_A",
            "timestamp": "2024-05-15T14:00",
            "temperature_c": 42.0,
            "air_pressure_mbar": 1005.0,
            "relative_humidity_pct": 20.0,  # Low RH as temperature warms
        })
        neighbors = [
            pd.Series({"station_id": "STN_B", "temperature_c": 41.5, "air_pressure_mbar": 1005.5, "relative_humidity_pct": 22.0}),
            pd.Series({"station_id": "STN_C", "temperature_c": 43.0, "air_pressure_mbar": 1004.8, "relative_humidity_pct": 19.0}),
        ]
        res = self.classifier.evaluate_observation(target_row, df_history=pd.DataFrame([target_row]), neighbor_rows=neighbors)

        self.assertEqual(res["classification"], "GENUINE_METEOROLOGICAL_EVENT")
        self.assertGreater(res["p_genuine_event"], 0.60)
        self.assertTrue(res["evidence"]["spatial_cooccurrence"])

    def test_isolated_sensor_spike(self):
        # Target station spikes to 45°C while neighbors are at 26°C and 27°C
        target_row = pd.Series({
            "station_id": "STN_A",
            "timestamp": "2024-01-15T14:00",
            "temperature_c": 45.0,
            "air_pressure_mbar": 1012.0,
            "relative_humidity_pct": 60.0,
        })
        neighbors = [
            pd.Series({"station_id": "STN_B", "temperature_c": 26.0, "air_pressure_mbar": 1012.0, "relative_humidity_pct": 60.0}),
            pd.Series({"station_id": "STN_C", "temperature_c": 27.0, "air_pressure_mbar": 1011.5, "relative_humidity_pct": 58.0}),
        ]
        res = self.classifier.evaluate_observation(target_row, df_history=pd.DataFrame([target_row]), neighbor_rows=neighbors)

        self.assertEqual(res["classification"], "HARDWARE_SENSOR_FAULT")
        self.assertGreater(res["p_sensor_fault"], 0.50)
        self.assertFalse(res["evidence"]["spatial_cooccurrence"])


class TestRootCauseClassifier(unittest.TestCase):
    """Verify multiclass root-cause classification."""

    def test_train_and_predict(self):
        # Synthetic mini-dataset with known labels
        records = []
        for i in range(15):
            records.append({
                "temperature_c": 25.0 + i * 0.1,
                "air_pressure_mbar": 1010.0,
                "relative_humidity_pct": 50.0,
                "anomaly_type": "NORMAL",
            })
            records.append({
                "temperature_c": 80.0,
                "air_pressure_mbar": 1010.0,
                "relative_humidity_pct": 50.0,
                "anomaly_type": "SPIKE",
            })
            records.append({
                "temperature_c": 25.0,
                "air_pressure_mbar": 1010.0,
                "relative_humidity_pct": 50.0,
                "anomaly_type": "FROZEN_SENSOR",
            })
            records.append({
                "temperature_c": np.nan,
                "air_pressure_mbar": 1010.0,
                "relative_humidity_pct": 50.0,
                "anomaly_type": "MISSING_DATA",
            })

        df = pd.DataFrame(records)
        classifier = RootCauseClassifier(n_estimators=20)
        classifier.fit(df, target_col="anomaly_type")

        preds, confs = classifier.predict(df)
        self.assertEqual(len(preds), len(df))
        self.assertTrue(np.all((confs >= 0.0) & (confs <= 1.0)))

        top_causes = classifier.get_top_causes(df.iloc[:5], top_k=2)
        self.assertEqual(len(top_causes), 5)
        self.assertEqual(len(top_causes[0]), 2)
        self.assertIn("cause", top_causes[0][0])
        self.assertIn("probability", top_causes[0][0])


if __name__ == "__main__":
    unittest.main()
