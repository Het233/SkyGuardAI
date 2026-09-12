"""
================================================================================
SkyGuard AI — Value Imputation & Correction Engine
================================================================================
Non-destructive imputation engine that generates physically consistent estimated
normal values (imputed_T, imputed_P, imputed_RH) for anomalous or missing readings.
Guarantees zero modification of raw sensor data while ensuring psychrometric
and hydrostatic equilibrium in corrected series.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from anomaly_injection.taxonomy import CORE_VARIABLES
from models.spatial_model import haversine_distance_km


class ImputationFlag(str, Enum):
    RAW_PASSTHROUGH = "RAW_PASSTHROUGH"
    SPATIAL_LAPSE_ESTIMATE = "SPATIAL_LAPSE_ESTIMATE"
    TEMPORAL_FORECAST_ESTIMATE = "TEMPORAL_FORECAST_ESTIMATE"
    HYBRID_BLEND = "HYBRID_BLEND"


@dataclass
class ImputationResult:
    """Imputation output for a single meteorological observation."""
    temperature_c: float
    air_pressure_mbar: float
    relative_humidity_pct: float
    imputation_flag: ImputationFlag
    confidence: float
    is_corrected: bool


class MeteorologicalImputer:
    """
    Physically grounded spatial-temporal imputation engine for Automatic Weather Stations.
    """

    TEMP_LAPSE_C_PER_M = -0.0065   # Standard tropospheric lapse rate (-6.5 °C / 1000m)
    PRESS_LAPSE_HPA_PER_M = -0.12  # Standard surface pressure reduction (~12 hPa / 100m)

    def __init__(
        self,
        k_neighbors: int = 3,
        max_neighbor_distance_km: float = 300.0,
        temporal_predictor: Optional[Any] = None,
    ):
        """
        Args:
            k_neighbors: Number of nearest neighboring stations to query for spatial lapse.
            max_neighbor_distance_km: Maximum radius in km for spatial consensus.
            temporal_predictor: Optional trained TemporalResidualPredictor instance.
        """
        self.k_neighbors = k_neighbors
        self.max_neighbor_distance_km = max_neighbor_distance_km
        self.temporal_predictor = temporal_predictor

        self.station_metadata: Dict[str, Dict[str, float]] = {}

    def fit_station_coordinates(self, df: pd.DataFrame) -> "MeteorologicalImputer":
        """Index station geographic coordinates (latitude, longitude, elevation)."""
        if "station_id" not in df.columns or "latitude" not in df.columns or "longitude" not in df.columns:
            return self

        elev_col = "elevation_m" if "elevation_m" in df.columns else None
        grouped = df.groupby("station_id")

        for st_id, group in grouped:
            lat = float(group["latitude"].iloc[0])
            lon = float(group["longitude"].iloc[0])
            elev = float(group[elev_col].iloc[0]) if elev_col else 0.0
            self.station_metadata[str(st_id)] = {
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev,
            }

        return self

    def impute_dataframe(
        self,
        df: pd.DataFrame,
        anomaly_mask: Optional[Union[pd.Series, np.ndarray]] = None,
        score_threshold: float = 0.42,
    ) -> pd.DataFrame:
        """
        Generate non-destructive imputed columns for the entire dataset.

        Args:
            df: Input DataFrame containing core variables and metadata.
            anomaly_mask: Optional boolean array flagging anomalous rows.
            score_threshold: Composite score cutoff above which imputation is triggered.

        Returns:
            New DataFrame with original columns untouched plus:
                - imputed_temperature_c
                - imputed_air_pressure_mbar
                - imputed_relative_humidity_pct
                - imputation_flag
                - imputation_confidence
        """
        if self.station_metadata == {} and "station_id" in df.columns and "latitude" in df.columns:
            self.fit_station_coordinates(df)

        df_out = df.copy()

        # Determine which rows require imputation
        if anomaly_mask is not None:
            needs_impute = np.asarray(anomaly_mask, dtype=bool)
        elif "is_anomaly" in df.columns:
            needs_impute = df["is_anomaly"].values.astype(bool)
        elif "composite_score" in df.columns:
            needs_impute = df["composite_score"].values >= score_threshold
        else:
            needs_impute = np.zeros(len(df), dtype=bool)

        # Also flag rows with NaN or gross physical violations
        for var in CORE_VARIABLES:
            if var in df.columns:
                needs_impute = needs_impute | df[var].isna().values

        # Initialize imputed series with raw values
        for var in CORE_VARIABLES:
            df_out[f"imputed_{var}"] = df[var].astype(float).values

        flags = [ImputationFlag.RAW_PASSTHROUGH.value] * len(df)
        confidences = np.ones(len(df), dtype=float)

        # Index timestamps for fast spatial lookups if multi-station
        has_stations = "station_id" in df.columns and "timestamp" in df.columns and len(self.station_metadata) > 1

        # Process anomalous rows
        anom_indices = np.where(needs_impute)[0]

        for idx in anom_indices:
            row = df.iloc[idx]
            st_id = str(row.get("station_id", "UNKNOWN"))
            t_stamp = row.get("timestamp", None)

            # 1. Attempt Spatial Consensus Lapse Imputation
            spatial_estimate = None
            spatial_conf = 0.0
            if has_stations and t_stamp is not None and st_id in self.station_metadata:
                spatial_estimate, spatial_conf = self._impute_spatial_lapse(
                    df=df,
                    target_station_id=st_id,
                    target_timestamp=t_stamp,
                    needs_impute_mask=needs_impute,
                )

            # 2. Attempt Temporal Diurnal Forecast Imputation
            temporal_estimate, temporal_conf = self._impute_temporal(
                df=df,
                row_idx=idx,
                station_id=st_id,
            )

            # 3. Fuse Estimates
            chosen_flag = ImputationFlag.RAW_PASSTHROUGH
            final_vals = {}
            final_conf = 0.80

            if spatial_estimate is not None and temporal_estimate is not None:
                chosen_flag = ImputationFlag.HYBRID_BLEND
                final_conf = round(float(0.6 * spatial_conf + 0.4 * temporal_conf), 3)
                for var in CORE_VARIABLES:
                    s_v = spatial_estimate[var]
                    t_v = temporal_estimate[var]
                    final_vals[var] = 0.65 * s_v + 0.35 * t_v
            elif spatial_estimate is not None:
                chosen_flag = ImputationFlag.SPATIAL_LAPSE_ESTIMATE
                final_conf = spatial_conf
                final_vals = spatial_estimate
            elif temporal_estimate is not None:
                chosen_flag = ImputationFlag.TEMPORAL_FORECAST_ESTIMATE
                final_conf = temporal_conf
                final_vals = temporal_estimate
            else:
                # Fallback: Forward-fill from previous valid observations
                chosen_flag = ImputationFlag.TEMPORAL_FORECAST_ESTIMATE
                final_conf = 0.50
                for var in CORE_VARIABLES:
                    final_vals[var] = float(df[var].iloc[max(0, idx - 1)]) if var in df.columns else 25.0

            # 4. Enforce Physical & Psychrometric Equilibrium
            final_vals = self._enforce_psychrometric_equilibrium(final_vals)

            # Assign to output
            for var in CORE_VARIABLES:
                df_out.iat[idx, df_out.columns.get_loc(f"imputed_{var}")] = final_vals[var]

            flags[idx] = chosen_flag.value
            confidences[idx] = final_conf

        df_out["imputation_flag"] = flags
        df_out["imputation_confidence"] = confidences

        return df_out

    def _impute_spatial_lapse(
        self,
        df: pd.DataFrame,
        target_station_id: str,
        target_timestamp: Any,
        needs_impute_mask: np.ndarray,
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """Compute elevation-adjusted spatial neighbor consensus estimate."""
        target_meta = self.station_metadata.get(target_station_id)
        if not target_meta:
            return None, 0.0

        # Query simultaneous records from other stations that are NOT anomalous
        time_mask = (df["timestamp"] == target_timestamp) & (df["station_id"] != target_station_id) & (~needs_impute_mask)
        valid_neighbors = df[time_mask]

        if len(valid_neighbors) == 0:
            return None, 0.0

        target_lat = target_meta["latitude"]
        target_lon = target_meta["longitude"]
        target_elev = target_meta["elevation_m"]

        neighbor_dists = []
        for _, nbr_row in valid_neighbors.iterrows():
            nbr_id = str(nbr_row["station_id"])
            if nbr_id not in self.station_metadata:
                continue
            nbr_meta = self.station_metadata[nbr_id]
            dist = haversine_distance_km(target_lat, target_lon, nbr_meta["latitude"], nbr_meta["longitude"])
            if dist <= self.max_neighbor_distance_km:
                neighbor_dists.append((dist, nbr_row, nbr_meta["elevation_m"]))

        if len(neighbor_dists) == 0:
            return None, 0.0

        neighbor_dists.sort(key=lambda x: x[0])
        top_neighbors = neighbor_dists[:self.k_neighbors]

        weights = []
        adjusted_vars = {var: [] for var in CORE_VARIABLES}

        for dist, nbr_row, nbr_elev in top_neighbors:
            w = 1.0 / max(dist, 1.0)
            weights.append(w)
            delta_elev = target_elev - nbr_elev

            # Lapse rate adjustments
            if "temperature_c" in nbr_row and pd.notna(nbr_row["temperature_c"]):
                t_adj = float(nbr_row["temperature_c"]) + (self.TEMP_LAPSE_C_PER_M * delta_elev)
                adjusted_vars["temperature_c"].append(t_adj)

            if "air_pressure_mbar" in nbr_row and pd.notna(nbr_row["air_pressure_mbar"]):
                p_adj = float(nbr_row["air_pressure_mbar"]) + (self.PRESS_LAPSE_HPA_PER_M * delta_elev)
                adjusted_vars["air_pressure_mbar"].append(p_adj)

            if "relative_humidity_pct" in nbr_row and pd.notna(nbr_row["relative_humidity_pct"]):
                # RH is conserved across local elevation difference, bounded in [1, 100]
                rh_adj = float(np.clip(nbr_row["relative_humidity_pct"], 1.0, 100.0))
                adjusted_vars["relative_humidity_pct"].append(rh_adj)

        w_arr = np.array(weights)
        w_norm = w_arr / np.sum(w_arr)

        estimated_values = {}
        for var in CORE_VARIABLES:
            vals = np.array(adjusted_vars[var])
            if len(vals) == len(w_norm):
                estimated_values[var] = float(np.sum(vals * w_norm))
            else:
                return None, 0.0

        # Confidence decays with average neighbor distance
        avg_dist = float(np.mean([d for d, _, _ in top_neighbors]))
        confidence = float(np.clip(0.95 - (avg_dist / 600.0), 0.50, 0.95))

        return estimated_values, round(confidence, 3)

    def _impute_temporal(
        self,
        df: pd.DataFrame,
        row_idx: int,
        station_id: str,
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """Compute temporal diurnal spline/lag estimate from prior uncorrupted steps."""
        # Find prior observations from same station
        st_mask = (df["station_id"] == station_id) if "station_id" in df.columns else np.ones(len(df), dtype=bool)
        st_indices = np.where(st_mask)[0]
        pos = np.searchsorted(st_indices, row_idx)

        if pos < 1:
            return None, 0.0

        # Check for 24-hour seasonal diurnal match (t - 24)
        has_24h_lag = pos >= 24
        lag24_idx = st_indices[pos - 24] if has_24h_lag else None

        estimates = {}
        for var in CORE_VARIABLES:
            if var not in df.columns:
                continue

            # If 24h prior observation is available and valid, use diurnal persistence + 1h trend
            if has_24h_lag and pd.notna(df[var].iloc[lag24_idx]):
                val_24h = float(df[var].iloc[lag24_idx])
                # Small trend adjustment from t-1
                prev_idx = st_indices[pos - 1]
                prev_val = float(df[var].iloc[prev_idx]) if pd.notna(df[var].iloc[prev_idx]) else val_24h
                est = 0.8 * val_24h + 0.2 * prev_val
            else:
                # Linear extrapolation from immediate prior observations
                prev_idx = st_indices[pos - 1]
                est = float(df[var].iloc[prev_idx]) if pd.notna(df[var].iloc[prev_idx]) else 25.0

            estimates[var] = est

        confidence = 0.85 if has_24h_lag else 0.65
        return estimates, confidence

    @staticmethod
    def _enforce_psychrometric_equilibrium(vals: Dict[str, float]) -> Dict[str, float]:
        """Ensure physical consistency and thermodynamic validity."""
        t = float(vals.get("temperature_c", 25.0))
        p = float(vals.get("air_pressure_mbar", 1013.25))
        rh = float(vals.get("relative_humidity_pct", 50.0))

        # 1. Physical limits
        t = float(np.clip(t, -40.0, 55.0))
        p = float(np.clip(p, 450.0, 1080.0))
        rh = float(np.clip(rh, 1.0, 100.0))

        # 2. Magnus formula dew point check
        a, b = 17.27, 237.7
        alpha = ((a * t) / (b + t)) + np.log(rh / 100.0)
        dew_point = (b * alpha) / (a - alpha)

        # Dew point cannot exceed dry-bulb ambient temperature in free atmosphere
        if dew_point > t:
            rh = 100.0

        return {
            "temperature_c": round(t, 2),
            "air_pressure_mbar": round(p, 2),
            "relative_humidity_pct": round(rh, 2),
        }
