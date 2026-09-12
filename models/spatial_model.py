"""
================================================================================
SkyGuard AI — Spatial Consensus Anomaly Detector
================================================================================
Leverages geographic spatial correlation across Automatic Weather Stations (AWS).
Compares Temperature, Pressure, and Humidity at target station against its nearest
neighbor stations, adjusting for elevation lapse rates.

Physical Constraints:
  - Operates strictly on Temperature, Pressure, and Relative Humidity.
  - Station coordinates (lat, lon, elevation) are metadata used for spatial indexing.
================================================================================
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from anomaly_injection.taxonomy import CORE_VARIABLES


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points on Earth in kilometers."""
    r_earth = 6371.0  # Earth's mean radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r_earth * c)


class SpatialConsensusDetector:
    """
    Multi-station spatial consensus detector.
    Identifies stations whose sensor readings deviate significantly from neighboring AWS.
    """

    # Atmospheric lapse rates for elevation normalization
    TEMP_LAPSE_RATE_C_PER_M = -0.0065   # Standard tropospheric lapse rate (-6.5 °C / 1000m)
    PRESS_LAPSE_RATE_HPA_PER_M = -0.12  # Standard surface pressure reduction (~12 hPa / 100m)

    def __init__(
        self,
        k_neighbors: int = 3,
        threshold_mad: float = 3.5,
        max_neighbor_distance_km: float = 300.0,
    ):
        """
        Args:
            k_neighbors: Number of nearest neighboring stations to consider (default: 3).
            threshold_mad: Median Absolute Deviation (MAD) cutoff for anomaly flag.
            max_neighbor_distance_km: Maximum search radius in km for neighbors.
        """
        self.k_neighbors = k_neighbors
        self.threshold_mad = threshold_mad
        self.max_neighbor_distance_km = max_neighbor_distance_km

        self.station_metadata: Dict[str, Dict[str, float]] = {}
        self.station_neighbors: Dict[str, List[str]] = {}

    def fit_station_topology(self, df: pd.DataFrame) -> "SpatialConsensusDetector":
        """
        Compute pairwise distances and neighbor mapping from station coordinates.
        """
        required_cols = ["station_id", "latitude", "longitude"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required spatial metadata column '{col}'.")

        unique_stations = df.drop_duplicates(subset=["station_id"])
        self.station_metadata = {}

        for _, row in unique_stations.iterrows():
            st_id = str(row["station_id"])
            self.station_metadata[st_id] = {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "elevation_m": float(row.get("elevation_m", 0.0)),
            }

        station_ids = list(self.station_metadata.keys())
        n_st = len(station_ids)
        self.station_neighbors = {}

        if n_st <= 1:
            for st in station_ids:
                self.station_neighbors[st] = []
            return self

        # Compute pairwise distance matrix
        for st1 in station_ids:
            lat1 = self.station_metadata[st1]["latitude"]
            lon1 = self.station_metadata[st1]["longitude"]

            distances = []
            for st2 in station_ids:
                if st1 == st2:
                    continue
                lat2 = self.station_metadata[st2]["latitude"]
                lon2 = self.station_metadata[st2]["longitude"]
                dist = haversine_distance_km(lat1, lon1, lat2, lon2)
                distances.append((st2, dist))

            # Sort by distance
            distances.sort(key=lambda x: x[1])
            # Select k nearest within max radius
            selected = [st for st, d in distances[:self.k_neighbors] if d <= self.max_neighbor_distance_km]
            # Fallback to closest regardless of distance if none within radius
            if not selected and distances:
                selected = [distances[0][0]]
            self.station_neighbors[st1] = selected

        return self

    def detect(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Evaluate spatial consensus across synchronous time steps.

        Returns:
            Tuple of:
                - is_spatial_anomaly: Binary Series (1 = spatial outlier, 0 = normal).
                - spatial_anomaly_score: Continuous score in [0.0, 1.0].
        """
        if not self.station_metadata:
            self.fit_station_topology(df)

        scores = pd.Series(0.0, index=df.index)
        flags = pd.Series(0, index=df.index)

        # Fast path if single station or no neighbors
        if not any(len(n) > 0 for n in self.station_neighbors.values()):
            return flags, scores

        # Process per timestamp across stations
        # Index dataframe by (timestamp, station_id) for rapid synchronous lookup
        has_timestamp = "timestamp" in df.columns
        if not has_timestamp:
            return flags, scores

        # Build pivot tables for the core variables
        pivots = {}
        for var in CORE_VARIABLES:
            if var in df.columns:
                pivots[var] = df.pivot_table(index="timestamp", columns="station_id", values=var)

        # Precompute elevation differences
        elevations = {st: self.station_metadata[st]["elevation_m"] for st in self.station_metadata}

        # Vectorized or row-level evaluation
        spatial_scores_df = pd.DataFrame(0.0, index=df.index, columns=CORE_VARIABLES)

        for var in CORE_VARIABLES:
            if var not in pivots:
                continue

            pv = pivots[var]
            for target_st, neighbors in self.station_neighbors.items():
                if not neighbors or target_st not in pv.columns:
                    continue

                target_vals = pv[target_st]
                target_elev = elevations.get(target_st, 0.0)

                # Collect neighbor values and apply elevation lapse correction
                neighbor_vals_list = []
                for n_st in neighbors:
                    if n_st in pv.columns:
                        n_val = pv[n_st].copy()
                        n_elev = elevations.get(n_st, 0.0)
                        elev_diff = target_elev - n_elev  # positive if target is higher

                        if var == "temperature_c":
                            n_val += elev_diff * self.TEMP_LAPSE_RATE_C_PER_M
                        elif var == "air_pressure_mbar":
                            n_val += elev_diff * self.PRESS_LAPSE_RATE_HPA_PER_M

                        neighbor_vals_list.append(n_val)

                if not neighbor_vals_list:
                    continue

                neighbor_mat = pd.concat(neighbor_vals_list, axis=1)
                neighbor_median = neighbor_mat.median(axis=1)

                # Robust dispersion: Median Absolute Deviation (MAD)
                abs_diff = neighbor_mat.sub(neighbor_median, axis=0).abs()
                mad = abs_diff.median(axis=1)
                # Scale MAD to approximate normal standard deviation (factor 1.4826)
                dispersion = (1.4826 * mad).clip(lower=0.5)

                residual = (target_vals - neighbor_median).abs()
                z_spatial = residual / dispersion

                # Normalize to [0, 1] relative to threshold
                norm_score = (z_spatial / (self.threshold_mad * 1.5)).clip(0.0, 1.0)

                # Map back to original dataframe rows
                st_rows = df[(df["station_id"] == target_st) & (df["timestamp"].isin(pv.index))].index
                row_timestamps = df.loc[st_rows, "timestamp"]
                spatial_scores_df.loc[st_rows, var] = norm_score.reindex(row_timestamps).values

        # Max score across the 3 core variables
        max_scores = spatial_scores_df.max(axis=1).fillna(0.0)
        threshold_norm = self.threshold_mad / (self.threshold_mad * 1.5)
        binary_flags = (max_scores >= threshold_norm).astype(int)

        return binary_flags, max_scores
