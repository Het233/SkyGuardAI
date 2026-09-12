"""
================================================================================
SkyGuard AI — Event-vs-Fault Reasoning Engine
================================================================================
Distinguishes genuine meteorological extreme events (heatwaves, cold fronts, squalls)
from faulty sensor hardware and communication breakdowns.

Operates strictly on the 3 core parameters: Temperature, Pressure, Relative Humidity.

Physical Principles:
  1. Multi-Station Spatial Co-occurrence:
     Natural weather phenomena (fronts, heatwaves) span regional scales (>50-100 km).
     If neighboring stations synchronously co-vary, the event is likely GENUINE.
     If an anomaly is isolated to a single station while neighbors remain calm, it is a SENSOR FAULT.
  2. Thermodynamic Coupling (Psychrometric Balance):
     Natural air warming causes relative humidity to drop (Clausius-Clapeyron relation).
     Contradictory departures (e.g. extreme thermal spike with saturated humidity) indicate SENSOR FAULT.
  3. Temporal Continuity & Transition Profiles:
     Meteorological fronts produce continuous time-series curves; ADC glitches or communication bit-flips
     produce single-step discontinuous spikes.
================================================================================
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from models.spatial_model import SpatialConsensusDetector
from anomaly_injection.taxonomy import CORE_VARIABLES


class EventVsFaultClassifier:
    """
    Reasoning engine attributing flagged anomalies to either genuine weather events
    or sensor hardware malfunctions.
    """

    def __init__(
        self,
        spatial_detector: Optional[SpatialConsensusDetector] = None,
        max_isolated_distance_km: float = 150.0,
    ):
        self.spatial_detector = spatial_detector or SpatialConsensusDetector(k_neighbors=3)
        self.max_isolated_distance_km = max_isolated_distance_km

    def evaluate_observation(
        self,
        row: pd.Series,
        df_history: pd.DataFrame,
        neighbor_rows: Optional[List[pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single observation for event-vs-fault reasoning.

        Returns:
            Dictionary containing:
                - p_genuine_event: float in [0.0, 1.0]
                - p_sensor_fault: float in [0.0, 1.0]
                - classification: 'GENUINE_METEOROLOGICAL_EVENT' or 'HARDWARE_SENSOR_FAULT'
                - evidence: checklist of physical verification indicators
                - reasoning_summary: human-interpretable explanation string
        """
        t = float(row.get("temperature_c", 25.0))
        p = float(row.get("air_pressure_mbar", 1013.0))
        rh = float(row.get("relative_humidity_pct", 50.0))

        # Check 1: Physical Boundary Check
        # If physically impossible on Earth, sensor fault probability is 1.0
        if t < -50.0 or t > 60.0 or p < 300.0 or p > 1100.0 or rh < 0.0 or rh > 100.0 or np.isnan(t) or np.isnan(p) or np.isnan(rh):
            return {
                "p_genuine_event": 0.01,
                "p_sensor_fault": 0.99,
                "classification": "HARDWARE_SENSOR_FAULT",
                "evidence": {
                    "physical_bounds_valid": False,
                    "spatial_cooccurrence": False,
                    "thermodynamic_coupling_valid": False,
                    "temporal_continuity": False,
                },
                "reasoning_summary": "Violates fundamental Earth atmospheric physical boundaries (sensor short/open circuit).",
            }

        # Check 2: Thermodynamic Psychrometric Coupling
        # Compute Magnus dew point spread
        a, b = 17.27, 237.7
        alpha = ((a * t) / (b + t)) + np.log(max(0.1, rh) / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        dew_point_spread = t - dew_point

        # Standard Earth dewpoint threshold (dew point > 32 °C is virtually non-existent in terrestrial atmosphere)
        thermo_valid = (dew_point_spread >= -0.5) and (dew_point <= 32.0)

        # Check 3: Spatial Co-occurrence
        spatial_agreement = False
        max_spatial_divergence = 0.0
        neighbor_agreement_count = 0

        if neighbor_rows and len(neighbor_rows) > 0:
            diffs = []
            for n_row in neighbor_rows:
                n_t = float(n_row.get("temperature_c", 25.0))
                diff = abs(t - n_t)
                diffs.append(diff)
                if diff < 6.0:
                    neighbor_agreement_count += 1
            if diffs:
                max_spatial_divergence = min(diffs)  # distance to closest neighbor
            if neighbor_agreement_count >= max(1, len(neighbor_rows) // 2):
                spatial_agreement = True

        # Check 4: Temporal Step Continuity (vs previous step if available)
        temporal_continuous = True
        st_id = row.get("station_id")
        current_time = row.get("timestamp")

        if st_id and current_time and "timestamp" in df_history.columns:
            st_sub = df_history[df_history["station_id"] == st_id]
            matching_positions = np.where(st_sub["timestamp"].values == current_time)[0]
            if len(matching_positions) > 0 and matching_positions[0] > 0:
                prev_row = st_sub.iloc[matching_positions[0] - 1]
                delta_t = abs(t - float(prev_row["temperature_c"]))
                delta_p = abs(p - float(prev_row["air_pressure_mbar"]))
                # Instantaneous discontinuous jump without neighbor support
                if (delta_t > 12.0 or delta_p > 15.0) and not spatial_agreement:
                    temporal_continuous = False

        # Evidence Scoring:
        # Genuine weather event probability increases with spatial agreement, thermodynamic coupling, and continuity
        fault_score = 0.0

        if not thermo_valid:
            fault_score += 0.40
        if neighbor_rows and len(neighbor_rows) > 0:
            if not spatial_agreement:
                if max_spatial_divergence > 10.0:
                    fault_score += 0.55
                else:
                    fault_score += 0.35
        if not temporal_continuous:
            fault_score += 0.25

        fault_score = min(0.99, max(0.01, fault_score))
        genuine_score = round(1.0 - fault_score, 3)
        fault_score = round(fault_score, 3)

        is_fault = (fault_score >= 0.50)
        classification = "HARDWARE_SENSOR_FAULT" if is_fault else "GENUINE_METEOROLOGICAL_EVENT"

        reasons = []
        if not spatial_agreement:
            reasons.append("Isolated reading divergent from synchronous neighboring stations")
        if not thermo_valid:
            reasons.append("Violates psychrometric temperature-humidity thermodynamic coupling")
        if not temporal_continuous:
            reasons.append("Discontinuous 1-step impulse incompatible with fluid atmospheric dynamics")
        if spatial_agreement and thermo_valid:
            reasons.append("Synchronous regional temperature and pressure shifts verified across neighborhood AWS")

        summary = "; ".join(reasons) if reasons else "Normal meteorological equilibrium."

        return {
            "p_genuine_event": genuine_score,
            "p_sensor_fault": fault_score,
            "classification": classification,
            "evidence": {
                "physical_bounds_valid": True,
                "spatial_cooccurrence": spatial_agreement,
                "thermodynamic_coupling_valid": thermo_valid,
                "temporal_continuity": temporal_continuous,
            },
            "reasoning_summary": summary,
        }

    def batch_classify(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Classify all flagged anomalies in a DataFrame.

        Returns:
            Tuple of:
                - p_genuine_event: pd.Series
                - p_sensor_fault: pd.Series
                - classification_series: pd.Series
                - evidence_summary_series: pd.Series
        """
        if not self.spatial_detector.station_metadata:
            self.spatial_detector.fit_station_topology(df)

        p_genuine = pd.Series(0.5, index=df.index)
        p_fault = pd.Series(0.5, index=df.index)
        class_res = pd.Series("NORMAL", index=df.index)
        summaries = pd.Series("", index=df.index)

        # Build pivot table for neighbor extraction
        pivots = {}
        for var in CORE_VARIABLES:
            if var in df.columns:
                pivots[var] = df.pivot_table(index="timestamp", columns="station_id", values=var)

        for idx, row in df.iterrows():
            st = str(row["station_id"])
            ts = str(row.get("timestamp", ""))

            # Find neighbors for this station at timestamp ts
            neighbors = self.spatial_detector.station_neighbors.get(st, [])
            neighbor_rows = []
            if neighbors and ts and "temperature_c" in pivots:
                for n_st in neighbors:
                    if n_st in pivots["temperature_c"].columns and ts in pivots["temperature_c"].index:
                        neighbor_rows.append({
                            "station_id": n_st,
                            "temperature_c": pivots["temperature_c"].loc[ts, n_st],
                            "air_pressure_mbar": pivots["air_pressure_mbar"].loc[ts, n_st] if "air_pressure_mbar" in pivots else 1010.0,
                            "relative_humidity_pct": pivots["relative_humidity_pct"].loc[ts, n_st] if "relative_humidity_pct" in pivots else 60.0,
                        })

            eval_res = self.evaluate_observation(row, df_history=df, neighbor_rows=neighbor_rows)

            p_genuine.loc[idx] = eval_res["p_genuine_event"]
            p_fault.loc[idx] = eval_res["p_sensor_fault"]
            class_res.loc[idx] = eval_res["classification"]
            summaries.loc[idx] = eval_res["reasoning_summary"]

        return p_genuine, p_fault, class_res, summaries
