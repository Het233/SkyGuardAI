"""
================================================================================
SkyGuard AI — Test Suite for Synthetic Anomaly Generator
================================================================================
"""

import unittest
import numpy as np
import pandas as pd

from anomaly_injection.taxonomy import (
    AnomalyType,
    AnomalySeverity,
    AnomalyEvent,
    CORE_VARIABLES,
)
from anomaly_injection.spikes import SpikeInjector
from anomaly_injection.freeze import FrozenSensorInjector
from anomaly_injection.drift import DriftInjector, ConstantOffsetInjector
from anomaly_injection.noise import NoiseInjector
from anomaly_injection.missing import MissingDataInjector
from anomaly_injection.communication import CommunicationCorruptionInjector
from anomaly_injection.physical import PhysicalBoundaryInjector
from anomaly_injection.multivariate import (
    CrossSensorInconsistencyInjector,
    CoordinatedAnomalyInjector,
)
from anomaly_injection.generator import SyntheticAnomalyGenerator


def create_synthetic_test_df(num_rows: int = 100, num_stations: int = 2) -> pd.DataFrame:
    """Helper to generate a clean mock AWS DataFrame for testing."""
    records = []
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    for s_idx in range(num_stations):
        st_id = f"TEST_AWS_{s_idx:04d}"
        for i in range(num_rows):
            t = base_time + pd.Timedelta(hours=i)
            # Simulating realistic diurnal cycle
            temp = 25.0 + 8.0 * np.sin(2 * np.pi * (i % 24) / 24.0)
            rh = 70.0 - 25.0 * np.sin(2 * np.pi * (i % 24) / 24.0)
            press = 1012.0 + 3.0 * np.cos(2 * np.pi * (i % 24) / 24.0)
            records.append({
                "station_id": st_id,
                "station_name": f"Station {s_idx}",
                "state": "TestState",
                "district": "TestDistrict",
                "latitude": 20.0 + s_idx,
                "longitude": 75.0 + s_idx,
                "elevation_m": 150.0,
                "timestamp": str(t),
                "temperature_c": temp,
                "air_pressure_mbar": press,
                "relative_humidity_pct": rh,
            })
    return pd.DataFrame(records)


class TestTaxonomy(unittest.TestCase):
    """Verify enums and metadata dataclass."""

    def test_anomaly_types(self):
        self.assertEqual(AnomalyType.SPIKE.value, "SPIKE")
        self.assertEqual(AnomalyType.FROZEN_SENSOR.value, "FROZEN_SENSOR")
        self.assertEqual(AnomalyType.SENSOR_DRIFT.value, "SENSOR_DRIFT")

    def test_anomaly_event_to_dict(self):
        ev = AnomalyEvent(
            event_id="EVT_TEST",
            station_id="TEST_AWS_0001",
            anomaly_type=AnomalyType.SPIKE,
            severity=AnomalySeverity.HIGH,
            target_variables=["temperature_c"],
            start_idx=10,
            end_idx=10,
            start_time="2024-01-01 10:00:00",
            end_time="2024-01-01 10:00:00",
            duration_steps=1,
            params={"magnitude": 15.0},
            description="Test spike",
        )
        d = ev.to_dict()
        self.assertEqual(d["event_id"], "EVT_TEST")
        self.assertEqual(d["anomaly_type"], "SPIKE")
        self.assertEqual(d["severity"], "HIGH")


class TestInjectors(unittest.TestCase):
    """Verify individual injector behaviors."""

    def setUp(self):
        self.df = create_synthetic_test_df(num_rows=100, num_stations=2)

    def test_spike_injector_positive(self):
        injector = SpikeInjector(seed=42)
        df_mod, events = injector.inject(
            self.df,
            target_variable="temperature_c",
            station_id="TEST_AWS_0000",
            start_idx=15,
            duration=1,
            direction="positive",
            magnitude=20.0,
        )
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.anomaly_type, AnomalyType.SPIKE)
        orig_val = self.df.loc[ev.start_idx, "temperature_c"]
        new_val = df_mod.loc[ev.start_idx, "temperature_c"]
        self.assertAlmostEqual(new_val, orig_val + 20.0, places=2)

    def test_spike_injector_negative(self):
        injector = SpikeInjector(seed=42)
        df_mod, events = injector.inject(
            self.df,
            target_variable="air_pressure_mbar",
            station_id="TEST_AWS_0000",
            start_idx=25,
            duration=1,
            direction="negative",
            magnitude=30.0,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.DROP)
        orig_val = self.df.loc[events[0].start_idx, "air_pressure_mbar"]
        new_val = df_mod.loc[events[0].start_idx, "air_pressure_mbar"]
        self.assertAlmostEqual(new_val, orig_val - 30.0, places=2)

    def test_frozen_sensor_injector(self):
        injector = FrozenSensorInjector(seed=42)
        duration = 8
        df_mod, events = injector.inject(
            self.df,
            target_variable="temperature_c",
            station_id="TEST_AWS_0000",
            start_idx=20,
            duration=duration,
        )
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.anomaly_type, AnomalyType.FROZEN_SENSOR)
        held_val = df_mod.loc[ev.start_idx, "temperature_c"]
        for idx in range(ev.start_idx, ev.end_idx + 1):
            self.assertEqual(df_mod.loc[idx, "temperature_c"], held_val)

    def test_drift_injector_linear(self):
        injector = DriftInjector(seed=42)
        duration = 20
        df_mod, events = injector.inject(
            self.df,
            target_variable="temperature_c",
            station_id="TEST_AWS_0000",
            start_idx=30,
            duration=duration,
            max_drift=10.0,
            drift_type="linear",
        )
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.anomaly_type, AnomalyType.SENSOR_DRIFT)
        # End should be drifted significantly from original
        orig_end = self.df.loc[ev.end_idx, "temperature_c"]
        new_end = df_mod.loc[ev.end_idx, "temperature_c"]
        self.assertNotEqual(orig_end, new_end)

    def test_constant_offset_injector(self):
        injector = ConstantOffsetInjector(seed=42)
        duration = 10
        df_mod, events = injector.inject(
            self.df,
            target_variable="air_pressure_mbar",
            station_id="TEST_AWS_0000",
            start_idx=40,
            duration=duration,
            offset=15.0,
        )
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.anomaly_type, AnomalyType.CONSTANT_OFFSET)
        for idx in range(ev.start_idx, ev.end_idx + 1):
            self.assertAlmostEqual(
                df_mod.loc[idx, "air_pressure_mbar"],
                self.df.loc[idx, "air_pressure_mbar"] + 15.0,
                places=2
            )

    def test_noise_injector(self):
        injector = NoiseInjector(seed=42)
        duration = 15
        df_mod, events = injector.inject(
            self.df,
            target_variable="temperature_c",
            station_id="TEST_AWS_0000",
            start_idx=50,
            duration=duration,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.HIGH_NOISE)

    def test_missing_data_injector(self):
        injector = MissingDataInjector(seed=42)
        duration = 5
        df_mod, events = injector.inject(
            self.df,
            target_variable="relative_humidity_pct",
            station_id="TEST_AWS_0000",
            start_idx=60,
            duration=duration,
            all_sensors=False,
        )
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.anomaly_type, AnomalyType.MISSING_DATA)
        for idx in range(ev.start_idx, ev.end_idx + 1):
            self.assertTrue(np.isnan(df_mod.loc[idx, "relative_humidity_pct"]))

    def test_communication_corruption_injector(self):
        injector = CommunicationCorruptionInjector(seed=42)
        df_mod, events = injector.inject(
            self.df,
            target_variable="temperature_c",
            station_id="TEST_AWS_0000",
            start_idx=70,
            corruption_mode="sign_flip",
        )
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.anomaly_type, AnomalyType.COMMUNICATION_CORRUPTION)
        orig_val = self.df.loc[ev.start_idx, "temperature_c"]
        self.assertAlmostEqual(df_mod.loc[ev.start_idx, "temperature_c"], -orig_val, places=2)

    def test_physical_boundary_injector(self):
        injector = PhysicalBoundaryInjector(seed=42)
        df_mod, events = injector.inject(
            self.df,
            target_variable="relative_humidity_pct",
            station_id="TEST_AWS_0000",
            start_idx=80,
            duration=2,
        )
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.anomaly_type, AnomalyType.PHYSICALLY_IMPOSSIBLE)
        for idx in range(ev.start_idx, ev.end_idx + 1):
            val = df_mod.loc[idx, "relative_humidity_pct"]
            self.assertTrue(val < 0.0 or val > 100.0)

    def test_cross_sensor_inconsistency_injector(self):
        injector = CrossSensorInconsistencyInjector(seed=42)
        df_mod, events = injector.inject(
            self.df,
            station_id="TEST_AWS_0000",
            start_idx=85,
            duration=2,
            inconsistency_mode="temp_rh_conflict",
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.CROSS_SENSOR_INCONSISTENCY)

    def test_coordinated_anomaly_injector(self):
        injector = CoordinatedAnomalyInjector(seed=42)
        df_mod, events = injector.inject(
            self.df,
            station_id="TEST_AWS_0000",
            start_idx=90,
            duration=2,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.COORDINATED_MULTIVARIATE)


class TestSyntheticAnomalyGenerator(unittest.TestCase):
    """Verify master generator execution and output schema."""

    def test_generator_full_pipeline(self):
        df = create_synthetic_test_df(num_rows=200, num_stations=2)
        gen = SyntheticAnomalyGenerator(anomaly_rate=0.08, seed=42)
        df_aug, events = gen.generate(df)

        # Check required columns exist
        expected_cols = [
            "is_anomaly",
            "anomaly_type",
            "anomaly_severity",
            "anomaly_target",
            "clean_temperature_c",
            "clean_air_pressure_mbar",
            "clean_relative_humidity_pct",
        ]
        for col in expected_cols:
            self.assertIn(col, df_aug.columns)

        # Verify pristine backup integrity
        for var in CORE_VARIABLES:
            pd.testing.assert_series_equal(
                df[var],
                df_aug[f"clean_{var}"],
                check_names=False
            )

        # Verify is_anomaly is binary
        unique_is_anomaly = set(df_aug["is_anomaly"].unique())
        self.assertTrue(unique_is_anomaly.issubset({0, 1}))

        # Verify ground truth consistency
        normal_mask = (df_aug["is_anomaly"] == 0)
        self.assertTrue((df_aug.loc[normal_mask, "anomaly_type"] == "NORMAL").all())

        anomaly_mask = (df_aug["is_anomaly"] == 1)
        self.assertTrue((df_aug.loc[anomaly_mask, "anomaly_type"] != "NORMAL").all())

        # Check summary report
        report = gen.get_summary_report(df_aug)
        self.assertGreater(report["total_anomalous_rows"], 0)
        self.assertGreater(report["total_events_injected"], 0)
        self.assertIn("distribution_by_type", report)


if __name__ == "__main__":
    unittest.main()
