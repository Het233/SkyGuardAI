"""
================================================================================
SkyGuard AI — Baseline Quality Control & Statistical Detectors Package
================================================================================
"""

from .quality_control import DeterministicQCValidator
from .statistical import (
    ZScoreDetector,
    IQRDetector,
    EWMADetector,
    CompositeBaselineDetector,
)

__all__ = [
    "DeterministicQCValidator",
    "ZScoreDetector",
    "IQRDetector",
    "EWMADetector",
    "CompositeBaselineDetector",
]
