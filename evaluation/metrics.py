"""
================================================================================
SkyGuard AI — Evaluation Metrics Harness
================================================================================
Standardized meteorological anomaly evaluation suite:
  - Precision, Recall, F1-Score, Specificity.
  - False Positive Rate (FPR) & False Alerts per Station-Day.
  - ROC-AUC & Precision-Recall AUC (PR-AUC).
  - Breakdown by Anomaly Type (Spike, Drift, Freeze, etc.).
  - Microsecond-level inference latency benchmarking.
"""

from typing import Dict, Any, List, Optional, Union
import time
import numpy as np
import pandas as pd

# Backward compatibility: np.trapezoid was added in NumPy 2.0; older versions have np.trapz
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def compute_binary_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_score: Optional[Union[np.ndarray, pd.Series]] = None,
    num_stations: int = 1,
    total_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics against binary ground truth.

    Args:
        y_true: Binary ground truth labels (0 = normal, 1 = anomaly).
        y_pred: Binary model predictions (0 = normal, 1 = anomaly).
        y_score: Optional continuous confidence / anomaly score in [0, 1].
        num_stations: Total number of stations in the evaluation set.
        total_hours: Total temporal span in hours for rate calculation.

    Returns:
        Dictionary of computed performance metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # False alerts per station-day
    if total_hours is not None and total_hours > 0 and num_stations > 0:
        station_days = (total_hours / 24.0) * num_stations
        false_alerts_per_station_day = fp / station_days if station_days > 0 else 0.0
    else:
        # Approximate: total rows / (num_stations * 24)
        approx_station_days = (total / max(1, num_stations)) / 24.0
        false_alerts_per_station_day = fp / approx_station_days if approx_station_days > 0 else 0.0

    # ROC-AUC & PR-AUC if continuous scores provided
    roc_auc = None
    pr_auc = None
    if y_score is not None:
        y_score = np.asarray(y_score).astype(float)
        roc_auc = _compute_roc_auc(y_true, y_score)
        pr_auc = _compute_pr_auc(y_true, y_score)

    return {
        "total_samples": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "false_alerts_per_station_day": round(false_alerts_per_station_day, 3),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
    }


def compute_type_breakdown(
    df: pd.DataFrame,
    y_pred_col: str,
    truth_type_col: str = "anomaly_type",
) -> Dict[str, Dict[str, Any]]:
    """
    Compute detection recall breakdown for each specific anomaly type.
    """
    breakdown = {}
    anomaly_types = df[truth_type_col].unique()

    for anom_type in anomaly_types:
        if str(anom_type).upper() == "NORMAL":
            continue
        mask = df[truth_type_col] == anom_type
        total_type = int(mask.sum())
        detected = int((mask & (df[y_pred_col] == 1)).sum())
        type_recall = detected / total_type if total_type > 0 else 0.0

        breakdown[str(anom_type).upper()] = {
            "total_instances": total_type,
            "detected": detected,
            "recall": round(type_recall, 4),
        }

    return breakdown


def _compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute Area Under ROC Curve via trapezoidal integration."""
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)
    if pos_count == 0 or neg_count == 0:
        return None

    # Sort descending by score
    desc_order = np.argsort(-y_score)
    y_true_sorted = y_true[desc_order]

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = tps / pos_count
    fpr = fps / neg_count

    # Prepend (0, 0)
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]

    return float(_trapezoid(tpr, fpr))


def _compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute Area Under Precision-Recall Curve (PR-AUC)."""
    pos_count = np.sum(y_true == 1)
    if pos_count == 0:
        return None

    desc_order = np.argsort(-y_score)
    y_true_sorted = y_true[desc_order]

    tps = np.cumsum(y_true_sorted == 1)
    total_preds = np.arange(1, len(y_true_sorted) + 1)

    precision = tps / total_preds
    recall = tps / pos_count

    # Prepend (1, 0)
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return float(_trapezoid(precision, recall))
