"""
================================================================================
SkyGuard AI — Statistical Anomaly Detectors
================================================================================
Implements time-series statistical anomaly detectors:
  1. Rolling Z-Score Detector (local mean & variance tracking).
  2. Rolling IQR / Tukey's Fences Detector (robust non-parametric outlier bounds).
  3. EWMA Residual Detector (exponential moving average tracking).
  4. Composite Baseline Detector (ensemble of deterministic rules + statistical tests).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from anomaly_injection.taxonomy import CORE_VARIABLES
from .quality_control import DeterministicQCValidator


class ZScoreDetector:
    """
    Rolling window Z-score outlier detector.
    Tracks local diurnal mean and standard deviation per station.
    """

    def __init__(self, window: int = 24, threshold: float = 3.5, min_std: float = 0.1):
        """
        Args:
            window: Rolling window length in steps (e.g. 24 for 24-hour diurnal cycle).
            threshold: Z-score cutoff for anomaly detection (default: 3.5).
            min_std: Lower bound on std to avoid zero-division in flat signals.
        """
        self.window = window
        self.threshold = threshold
        self.min_std = min_std

    def detect(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling Z-scores across all core variables per station.

        Returns:
            Tuple of:
                - is_anomaly: Binary Series (1 if any core variable exceeds threshold, 0 otherwise).
                - anomaly_score: Continuous score in [0.0, 1.0].
        """
        scores = pd.DataFrame(index=df.index)
        flags = pd.DataFrame(index=df.index)

        stations = df["station_id"].unique() if "station_id" in df.columns else ["ALL"]

        for st in stations:
            st_mask = (df["station_id"] == st) if "station_id" in df.columns else pd.Series(True, index=df.index)
            st_indices = df.index[st_mask]

            for var in CORE_VARIABLES:
                if var not in df.columns:
                    continue

                series = df.loc[st_indices, var]
                # Rolling mean and std (centered or backward-looking)
                r_mean = series.rolling(window=self.window, min_periods=min(6, self.window)).mean()
                r_std = series.rolling(window=self.window, min_periods=min(6, self.window)).std()

                # Fallback to series global stats for early warm-up periods
                r_mean = r_mean.fillna(series.mean())
                r_std = r_std.fillna(series.std()).clip(lower=self.min_std)

                z_val = (series - r_mean).abs() / r_std
                # Normalize score to [0, 1] relative to threshold
                norm_score = (z_val / (self.threshold * 1.5)).clip(0.0, 1.0)

                scores.loc[st_indices, var] = norm_score
                flags.loc[st_indices, var] = (z_val > self.threshold)

        max_score = scores.max(axis=1).fillna(0.0)
        any_flag = flags.any(axis=1).fillna(False).astype(int)

        return any_flag, max_score


class IQRDetector:
    """
    Robust non-parametric outlier detector based on rolling Interquartile Range (Tukey's Fences).
    """

    def __init__(self, window: int = 48, multiplier: float = 1.8, min_iqr: float = 0.2):
        """
        Args:
            window: Rolling window size (default: 48 hours for 2-day diurnal stability).
            multiplier: IQR fence multiplier (default: 1.8; standard Tukey uses 1.5).
            min_iqr: Minimum IQR floor to avoid false alarms in invariant intervals.
        """
        self.window = window
        self.multiplier = multiplier
        self.min_iqr = min_iqr

    def detect(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling IQR fences across core variables.

        Returns:
            Tuple of (binary_flag_series, continuous_score_series).
        """
        scores = pd.DataFrame(index=df.index)
        flags = pd.DataFrame(index=df.index)

        stations = df["station_id"].unique() if "station_id" in df.columns else ["ALL"]

        for st in stations:
            st_mask = (df["station_id"] == st) if "station_id" in df.columns else pd.Series(True, index=df.index)
            st_indices = df.index[st_mask]

            for var in CORE_VARIABLES:
                if var not in df.columns:
                    continue

                series = df.loc[st_indices, var]
                r_q25 = series.rolling(window=self.window, min_periods=min(12, self.window)).quantile(0.25)
                r_q75 = series.rolling(window=self.window, min_periods=min(12, self.window)).quantile(0.75)

                r_q25 = r_q25.fillna(series.quantile(0.25))
                r_q75 = r_q75.fillna(series.quantile(0.75))

                iqr = (r_q75 - r_q25).clip(lower=self.min_iqr)
                lower_fence = r_q25 - self.multiplier * iqr
                upper_fence = r_q75 + self.multiplier * iqr

                is_outlier = (series < lower_fence) | (series > upper_fence)

                # Distance outside fence normalized by IQR
                dist_below = (lower_fence - series).clip(lower=0.0)
                dist_above = (series - upper_fence).clip(lower=0.0)
                excess = dist_below + dist_above
                norm_score = (excess / (self.multiplier * iqr)).clip(0.0, 1.0)

                scores.loc[st_indices, var] = norm_score
                flags.loc[st_indices, var] = is_outlier

        max_score = scores.max(axis=1).fillna(0.0)
        any_flag = flags.any(axis=1).fillna(False).astype(int)

        return any_flag, max_score


class EWMADetector:
    """
    Exponentially Weighted Moving Average (EWMA) detector for tracking dynamic baseline drift.
    """

    def __init__(self, span: int = 24, sigma_threshold: float = 3.5):
        self.span = span
        self.sigma_threshold = sigma_threshold

    def detect(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Compute EWMA residuals across core variables.

        Returns:
            Tuple of (binary_flag_series, continuous_score_series).
        """
        scores = pd.DataFrame(index=df.index)
        flags = pd.DataFrame(index=df.index)

        stations = df["station_id"].unique() if "station_id" in df.columns else ["ALL"]

        for st in stations:
            st_mask = (df["station_id"] == st) if "station_id" in df.columns else pd.Series(True, index=df.index)
            st_indices = df.index[st_mask]

            for var in CORE_VARIABLES:
                if var not in df.columns:
                    continue

                series = df.loc[st_indices, var]
                prior = series.shift(1)
                ewma_mean = prior.ewm(span=self.span, adjust=False).mean().fillna(series)
                ewma_std = prior.ewm(span=self.span, adjust=False).std().fillna(series.std()).clip(lower=0.1)

                residual = (series - ewma_mean).abs()
                z_ewma = residual / ewma_std

                norm_score = (z_ewma / (self.sigma_threshold * 1.5)).clip(0.0, 1.0)
                scores.loc[st_indices, var] = norm_score
                flags.loc[st_indices, var] = (z_ewma > self.sigma_threshold)

        max_score = scores.max(axis=1).fillna(0.0)
        any_flag = flags.any(axis=1).fillna(False).astype(int)

        return any_flag, max_score


class CompositeBaselineDetector:
    """
    Ensemble baseline detector integrating:
      1. Deterministic QC (Physical bounds, rate-of-change, persistence flatlines, NaNs).
      2. Rolling Z-Score.
      3. Rolling IQR.
      4. EWMA Residuals.
    """

    def __init__(
        self,
        z_threshold: float = 3.5,
        iqr_multiplier: float = 1.8,
        ewma_threshold: float = 3.5,
    ):
        self.qc = DeterministicQCValidator()
        self.z_detector = ZScoreDetector(threshold=z_threshold)
        self.iqr_detector = IQRDetector(multiplier=iqr_multiplier)
        self.ewma_detector = EWMADetector(sigma_threshold=ewma_threshold)

    def detect(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Run composite baseline detection.

        Returns:
            Tuple of:
                - composite_flags: Binary Series (1 = anomaly, 0 = normal).
                - composite_scores: Continuous Series in [0.0, 1.0].
                - component_table: DataFrame containing individual detector predictions.
        """
        qc_flag, qc_details = self.qc.validate(df)
        z_flag, z_score = self.z_detector.detect(df)
        iqr_flag, iqr_score = self.iqr_detector.detect(df)
        ewma_flag, ewma_score = self.ewma_detector.detect(df)

        component_table = pd.DataFrame(index=df.index)
        component_table["qc_flag"] = qc_flag.astype(int)
        component_table["z_flag"] = z_flag.astype(int)
        component_table["iqr_flag"] = iqr_flag.astype(int)
        component_table["ewma_flag"] = ewma_flag.astype(int)

        component_table["z_score"] = z_score
        component_table["iqr_score"] = iqr_score
        component_table["ewma_score"] = ewma_score

        # Composite score: if deterministic QC fails, score is 1.0; otherwise max of statistical scores
        stat_max = pd.concat([z_score, iqr_score, ewma_score], axis=1).max(axis=1)
        composite_score = np.where(qc_flag, 1.0, stat_max)
        composite_score = pd.Series(composite_score, index=df.index)

        # Flagged if QC fails OR at least one statistical detector flags
        composite_flag = (qc_flag | (z_flag == 1) | (iqr_flag == 1) | (ewma_flag == 1)).astype(int)

        return composite_flag, composite_score, component_table
