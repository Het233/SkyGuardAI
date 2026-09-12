# SkyGuard AI — End-to-End Pipeline Demonstration Report

- **Total Ingested Observations:** `15,000`
- **Ground-Truth Anomalies:** `687`
- **Imputed / Corrected Records:** `687`

## 1. Station Health Status Overview

| Station ID | Overall Health (0-100) | Status | Lowest Sensor | Operational Urgency |
| :--- | :---: | :---: | :---: | :---: |
| `IMD_AWS_0001` | **77.8** | `GOOD` | `relative_humidity_pct` | `ROUTINE` |
| `IMD_AWS_0002` | **96.3** | `EXCELLENT` | `relative_humidity_pct` | `ROUTINE` |
| `IMD_AWS_0003` | **28.6** | `CRITICAL` | `relative_humidity_pct` | `EMERGENCY_DISPATCH` |
| `IMD_AWS_0005` | **55.0** | `DEGRADING` | `relative_humidity_pct` | `SCHEDULED_MAINTENANCE` |
| `IMD_AWS_0006` | **46.3** | `CRITICAL` | `air_pressure_mbar` | `EMERGENCY_DISPATCH` |
| `IMD_AWS_0007` | **83.3** | `GOOD` | `relative_humidity_pct` | `ROUTINE` |
| `IMD_AWS_0008` | **77.0** | `GOOD` | `air_pressure_mbar` | `ROUTINE` |
| `IMD_AWS_0009` | **78.7** | `GOOD` | `air_pressure_mbar` | `ROUTINE` |
| `IMD_AWS_0010` | **34.0** | `CRITICAL` | `relative_humidity_pct` | `EMERGENCY_DISPATCH` |
| `IMD_AWS_0011` | **66.8** | `DEGRADING` | `temperature_c` | `SCHEDULED_MAINTENANCE` |

---

## 2. Non-Destructive Imputation Performance

| Meteorological Parameter | Corrupted Raw MAE | Imputed Corrected MAE | Error Reduction |
| :--- | :---: | :---: | :---: |
| `temperature_c` | 2.98 | **0.906** | **69.6%** |
| `air_pressure_mbar` | 307.372 | **7.915** | **97.4%** |
| `relative_humidity_pct` | 8.186 | **5.815** | **29.0%** |

---

## 3. Sample Diagnostic Alert Card

# ⚠️ SkyGuard AI Diagnostic Alert — Station IMD_AWS_0001
**Timestamp:** `2023-01-01 06:00:00` | **Severity:** `CRITICAL` | **Score:** `1.000`

## 🔍 Diagnostic Summary
- **Attributed Failure Mode:** `COORDINATED_MULTIVARIATE` (Multi-Sensor Coordinated Failure / Power Bus Failure)
- **Confidence:** `81.0%`
- **Primary Sensor Culprit:** `temperature_c`
- **Maintenance Urgency:** `CRITICAL`

## 📋 Evidence & Physical Discrepancies
- Rate-of-change limit exceeded on 'temperature_c': Δ=-13.86 in 1 hr (limit: 12.0).
- Rate-of-change limit exceeded on 'air_pressure_mbar': Δ=+33.68 in 1 hr (limit: 15.0).
- Persistence check: 'relative_humidity_pct' identical to previous reading (Δ=0.00).
- temperature_c diverged by -14.66 from 9 neighbors (MAD Z: 14.1); air_pressure_mbar diverged by +34.28 from 9 neighbors (MAD Z: 6.4)
- **Spatial Neighbor Disparity:** temperature_c diverged by -14.66 from 9 neighbors (MAD Z: 14.1); air_pressure_mbar diverged by +34.28 from 9 neighbors (MAD Z: 6.4)

## 🛠️ Prescriptive Maintenance SOP
> **Standard Protocol:** `IMD-AWS-STD-SOP-V4`
> **Primary Action:** Station power rail instability affecting all channels simultaneously. Immediate power supply diagnostic required.
> **Field Technician Task:** Replace 12V solar buffer battery and test 5V/3.3V internal voltage regulators on AWS mainboard.
