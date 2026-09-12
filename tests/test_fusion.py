"""
================================================================================
SkyGuard AI — Test Suite for Spatial Consensus & Hybrid Fusion
================================================================================
"""

import unittest
import numpy as np
import pandas as pd

from models.spatial_model import SpatialConsensusDetector, haversine_distance_km
from fusion.engine import HybridAnomalyFusionEngine


class TestSpatialConsensus(unittest.TestCase):
    """Verify spatial distance, topology mapping, and consensus outlier detection."""

    def test_haversine_distance(self):
        # Distance between Mumbai (19.0760, 72.8777) and Pune (18.5204, 73.8567) is ~120 km
        dist = haversine_distance_km(19.0760, 72.8777, 18.5204, 73.8567)
        self.assertAlmostEqual(dist, 120.0, delta=20.0)

    def test_spatial_topology_and_detection(self):
        # Create 3 nearby stations with synchronous timestamps
        records = []
        timestamps = [f"2024-01-01T{h:02d}:00" for h in range(10)]
        stations = [
            {"id": "STN_A", "lat": 16.0, "lon": 80.0, "elev": 30.0},
            {"id": "STN_B", "lat": 16.1, "lon": 80.1, "elev": 35.0},
            {"id": "STN_C", "lat": 16.05, "lon": 80.05, "elev": 32.0},
        ]

        for s in stations:
            for t in timestamps:
                # Normal synchronous weather
                records.append({
                    "station_id": s["id"],
                    "latitude": s["lat"],
                    "longitude": s["lon"],
                    "elevation_m": s["elev"],
                    "timestamp": t,
                    "temperature_c": 28.0,
                    "air_pressure_mbar": 1010.0,
                    "relative_humidity_pct": 60.0,
                })

        df = pd.DataFrame(records)

        # Inject spatial outlier at step 5 for STN_A only (neighbor stations remain at 28°C)
        mask_target = (df["station_id"] == "STN_A") & (df["timestamp"] == timestamps[5])
        df.loc[mask_target, "temperature_c"] = 55.0

        detector = SpatialConsensusDetector(k_neighbors=2, threshold_mad=3.0)
        detector.fit_station_topology(df)

        self.assertIn("STN_A", detector.station_neighbors)
        self.assertIn("STN_B", detector.station_neighbors["STN_A"])

        flags, scores = detector.detect(df)

        # STN_A at step 5 should be flagged as spatial outlier
        target_idx = df[mask_target].index[0]
        self.assertEqual(flags.loc[target_idx], 1)
        self.assertGreater(scores.loc[target_idx], 0.7)

        # Neighbor stations at step 5 should NOT be flagged
        mask_stn_b = (df["station_id"] == "STN_B") & (df["timestamp"] == timestamps[5])
        idx_b = df[mask_stn_b].index[0]
        self.assertEqual(flags.loc[idx_b], 0)


class TestHybridFusionEngine(unittest.TestCase):
    """Verify calibrated multi-model score fusion and severity classifications."""

    def setUp(self):
        self.engine = HybridAnomalyFusionEngine(decision_threshold=0.45)

    def test_fuse_scores_normal(self):
        n = 5
        qc_flags = np.zeros(n)
        stat_scores = np.full(n, 0.1)
        iso_scores = np.full(n, 0.15)
        ae_scores = np.full(n, 0.08)
        temp_scores = np.full(n, 0.05)
        spatial_scores = np.full(n, 0.02)

        composite, preds, conf, sev = self.engine.fuse_scores(
            qc_flags, stat_scores, iso_scores, ae_scores, temp_scores, spatial_scores
        )

        self.assertTrue(np.all(composite < 0.35))
        self.assertTrue(np.all(preds == 0))
        self.assertTrue(np.all(sev == "NONE"))

    def test_fuse_scores_qc_override(self):
        # If deterministic QC triggers, composite score must be 1.0 (Critical)
        qc_flags = np.array([1.0, 0.0])
        stat_scores = np.array([0.2, 0.1])
        iso_scores = np.array([0.1, 0.1])
        ae_scores = np.array([0.2, 0.1])
        temp_scores = np.array([0.1, 0.1])

        composite, preds, conf, sev = self.engine.fuse_scores(
            qc_flags, stat_scores, iso_scores, ae_scores, temp_scores
        )

        self.assertEqual(composite[0], 1.0)
        self.assertEqual(preds[0], 1)
        self.assertEqual(sev[0], "CRITICAL")

    def test_consensus_confidence(self):
        # When 4 of 5 active models agree, confidence should be 0.8
        qc_flags = np.array([0.0])
        stat_scores = np.array([0.8])
        iso_scores = np.array([0.7])
        ae_scores = np.array([0.9])
        temp_scores = np.array([0.6])
        spatial_scores = np.array([0.1])

        composite, preds, conf, sev = self.engine.fuse_scores(
            qc_flags, stat_scores, iso_scores, ae_scores, temp_scores, spatial_scores
        )

        self.assertEqual(preds[0], 1)
        # 4 out of 6 active detectors >= 0.5 (stat, iso, ae, temp)
        self.assertAlmostEqual(conf[0], 4 / 6, places=2)


if __name__ == "__main__":
    unittest.main()
