# SkyGuard AI — Baseline Detectors Benchmark Report

**Dataset:** `data/synthetic/sample_synthetic_anomalies.csv`  
**Observations:** 87,600 rows | **Stations:** 10 | **Total Anomalies:** 4,471 (5.10%)

## Performance Comparison

| Method | Precision | Recall | F1-Score | PR-AUC | False Positive Rate | False Alerts/Stn-Day | Latency/Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic QC** | 0.8118 | 0.4603 | 0.5875 | 0.4208 | 0.57% | 1.31 | 3.0 µs |
| **Rolling Z-Score** | 0.7397 | 0.0483 | 0.0907 | 0.1614 | 0.09% | 0.21 | 1.6 µs |
| **Rolling IQR** | 0.5666 | 0.2196 | 0.3166 | 0.2244 | 0.90% | 2.06 | 3.5 µs |
| **EWMA Residuals** | 0.7690 | 0.0588 | 0.1093 | 0.1487 | 0.10% | 0.22 | 1.6 µs |
| **Composite Baseline** | 0.6692 | 0.5833 | 0.6233 | 0.5144 | 1.55% | 3.53 | 9.5 µs |

## Composite Baseline Recall by Anomaly Type

| Anomaly Type | Injected Instances | Detected Instances | Detection Recall |
| :--- | :---: | :---: | :---: |
| `FROZEN_SENSOR` | 928 | 928 | **100.0%** |
| `PHYSICALLY_IMPOSSIBLE` | 20 | 20 | **100.0%** |
| `MISSING_DATA` | 147 | 147 | **100.0%** |
| `COORDINATED_MULTIVARIATE` | 34 | 34 | **100.0%** |
| `COMMUNICATION_CORRUPTION` | 20 | 20 | **100.0%** |
| `DROP` | 44 | 41 | **93.2%** |
| `CROSS_SENSOR_INCONSISTENCY` | 26 | 24 | **92.3%** |
| `SPIKE` | 54 | 44 | **81.5%** |
| `HIGH_NOISE` | 462 | 284 | **61.5%** |
| `CONSTANT_OFFSET` | 809 | 469 | **58.0%** |
| `SENSOR_DRIFT` | 1927 | 597 | **31.0%** |

## Key Findings & Research Insights
1. **Deterministic QC** excels at gross errors (Physical Bounds, NaNs, Frozen Sensors, Spikes) with near-zero false alarms.
2. **Statistical Detectors (Z-score & IQR)** catch dynamic departures but have difficulty distinguishing gradual drift from seasonal cycles.
3. **Composite Baseline** achieves a strong baseline foundation (**F1 benchmark**) for the Machine Learning models (Isolation Forest, Autoencoder, and Residual Predictors) to improve upon in Phase 4.
