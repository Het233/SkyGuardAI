"""
================================================================================
SkyGuard AI — Meteorological Feature Engineering Pipeline
================================================================================
Extracts temporal lags, rolling moments, cyclical diurnal/seasonal harmonics,
and thermodynamic coupling features strictly from Temperature, Pressure, and Humidity.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from anomaly_injection.taxonomy import CORE_VARIABLES


class WeatherFeatureEngineer:
    """
    Feature engineering pipeline for AWS time-series observations.
    Maintains strict physical constraints: operates solely on T, P, and RH.
    """

    def __init__(
        self,
        lag_steps: Tuple[int, ...] = (1, 2, 6, 12, 24),
        rolling_windows: Tuple[int, ...] = (6, 24),
        include_cyclical: bool = True,
        include_thermodynamic: bool = True,
    ):
        self.lag_steps = lag_steps
        self.rolling_windows = rolling_windows
        self.include_cyclical = include_cyclical
        self.include_thermodynamic = include_thermodynamic

        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def _extract_cyclical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cyclical Fourier embeddings for hour of day and day of year."""
        feat_df = pd.DataFrame(index=df.index)

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            hours = ts.dt.hour.fillna(12).values
            day_of_year = ts.dt.dayofyear.fillna(180).values
        else:
            # Fallback to integer step index modulo 24
            hours = (np.arange(len(df)) % 24)
            day_of_year = ((np.arange(len(df)) // 24) % 365) + 1

        feat_df["sin_hour"] = np.sin(2.0 * np.pi * hours / 24.0)
        feat_df["cos_hour"] = np.cos(2.0 * np.pi * hours / 24.0)
        feat_df["sin_doy"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
        feat_df["cos_doy"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

        return feat_df

    def _extract_thermodynamic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract psychrometric & thermodynamic approximations using only T, P, RH.
        Uses the Magnus-Tetens approximation for dew point and saturation vapor pressure.
        """
        feat_df = pd.DataFrame(index=df.index)

        t = df["temperature_c"].values
        p = df["air_pressure_mbar"].values
        rh = np.clip(df["relative_humidity_pct"].values, 1.0, 100.0)

        # Magnus parameterization constants
        a = 17.27
        b = 237.7
        alpha = ((a * t) / (b + t)) + np.log(rh / 100.0)
        # Estimated dew point (°C)
        dew_point = (b * alpha) / (a - alpha)

        # Dew point depression (difference between ambient temperature and dew point)
        feat_df["dew_point_spread"] = t - dew_point

        # Saturation vapor pressure proxy (hPa)
        sat_vp = 6.112 * np.exp((17.67 * t) / (t + 243.5))
        # Actual vapor pressure proxy (hPa)
        feat_df["vapor_pressure_proxy"] = sat_vp * (rh / 100.0)

        # Gas law specific volume proxy (Kelvin / mbar)
        feat_df["gas_law_density_proxy"] = (t + 273.15) / np.clip(p, 300.0, 1150.0)

        # Thermal-moisture interaction
        feat_df["temp_rh_interaction"] = (t / 40.0) * (rh / 100.0)

        return feat_df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all tabular and temporal features per station.

        Args:
            df: Input AWS DataFrame containing station_id and core variables.

        Returns:
            pd.DataFrame of engineered features with no NaNs.
        """
        feature_blocks = []

        # 1. Base Core Variables
        base_features = df[CORE_VARIABLES].copy()
        feature_blocks.append(base_features)

        # 2. Station-Isolated Temporal Lags and Rolling Moments
        temporal_dfs = []
        stations = df["station_id"].unique() if "station_id" in df.columns else ["ALL"]

        for st in stations:
            st_mask = (df["station_id"] == st) if "station_id" in df.columns else pd.Series(True, index=df.index)
            st_sub = df.loc[st_mask, CORE_VARIABLES].copy()
            st_feat = pd.DataFrame(index=st_sub.index)

            for var in CORE_VARIABLES:
                series = st_sub[var]

                # Lags
                for lag in self.lag_steps:
                    st_feat[f"{var}_lag_{lag}"] = series.shift(lag)

                # Differences (Rate of change and acceleration)
                diff1 = series.diff()
                st_feat[f"{var}_diff_1"] = diff1
                st_feat[f"{var}_accel"] = diff1.diff()

                # Rolling statistics
                for w in self.rolling_windows:
                    r = series.rolling(window=w, min_periods=min(3, w))
                    st_feat[f"{var}_roll_mean_{w}"] = r.mean()
                    st_feat[f"{var}_roll_std_{w}"] = r.std()
                    st_feat[f"{var}_roll_range_{w}"] = r.max() - r.min()

            # Backfill initial warm-up NaNs with first available observation
            st_feat = st_feat.bfill().ffill()
            temporal_dfs.append(st_feat)

        combined_temporal = pd.concat(temporal_dfs).loc[df.index]
        feature_blocks.append(combined_temporal)

        # 3. Cyclical Diurnal / Seasonal Features
        if self.include_cyclical:
            cyclical_df = self._extract_cyclical(df)
            feature_blocks.append(cyclical_df)

        # 4. Thermodynamic Features
        if self.include_thermodynamic:
            thermo_df = self._extract_thermodynamic(df)
            feature_blocks.append(thermo_df)

        full_features = pd.concat(feature_blocks, axis=1)

        # Clean any remaining NaNs with column medians
        full_features = full_features.fillna(full_features.median())
        # If any column is entirely NaN, fill with 0.0
        full_features = full_features.fillna(0.0)

        return full_features

    def fit(self, df: pd.DataFrame) -> "WeatherFeatureEngineer":
        """
        Extract features on training data and fit the StandardScaler.
        """
        X = self.extract_features(df)
        self.feature_names = list(X.columns)
        self.scaler = StandardScaler()
        self.scaler.fit(X.values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and scale features using the fitted scaler.
        """
        if self.scaler is None:
            raise RuntimeError("WeatherFeatureEngineer must be fitted before calling transform().")
        X = self.extract_features(df)
        # Ensure column ordering matches fitted state
        X = X[self.feature_names]
        return self.scaler.transform(X.values)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in a single call."""
        return self.fit(df).transform(df)
