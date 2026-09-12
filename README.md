# 🛰️ SkyGuard AI

**Intelligent Anomaly Detection & Predictive Sensor Health Monitoring for Automatic Weather Stations (AWS)**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Open-Meteo](https://img.shields.io/badge/Data%20Source-Open--Meteo-orange.svg)](https://open-meteo.com/)

---

## 📌 Overview

**SkyGuard AI** is an end-to-end meteorological sensor surveillance system engineered to detect real-time anomalies, sensor degradation, and communication failures in Automatic Weather Stations (AWS).

To preserve realistic physical constraints and operational viability across varied weather networks, SkyGuard AI operates strictly on the primary triumvirate of core atmospheric variables:
1. **Temperature (°C)** (`temperature_2m`)
2. **Surface Air Pressure (hPa / mbar)** (`surface_pressure`)
3. **Relative Humidity (%)** (`relative_humidity_2m`)

Spatial comparison and neighborhood consistency checks leverage the identical trio of variables from adjacent weather stations.

---

## 🏛️ System Architecture

```text
               AWS Sensor Network
  (Temperature, Surface Pressure, Relative Humidity)
                       │
                       ▼
              Data Ingestion Layer
       (Streaming API / CSV ingestion)
                       │
                       ▼
            Deterministic Validation
   (Physical plausibility, rate of change, missingness)
                       │
                       ▼
              Feature Engineering
  (Lags, rolling stats, diurnal cycles, cross-variable)
                       │
                       ▼
              AI Anomaly Engine
 ┌─────────────────────┬─────────────────────┐
 │ Statistical (IQR/Z) │   Isolation Forest  │
 ├─────────────────────┼─────────────────────┤
 │ Autoencoder (Recon) │ Temporal Residuals  │
 └─────────────────────┴─────────────────────┘
                       │
                       ▼
             Anomaly Fusion Layer
     (Consensus scoring, severity & confidence)
                       │
                       ▼
             Explainability (XAI)
         (SHAP values & root-cause clues)
                       │
                       ▼
          Sensor Health & Alerts Dashboard
```

---

## 🚀 Key Capabilities

- **Pan-India Weather Station Coverage**: Master registry and ingestion pipeline supporting 826 Automatic Weather Stations covering all 36 Indian States and Union Territories.
- **Fault Taxonomy Detection**: Identifies spikes, sensor freezes, gradual sensor drift, calibration offsets, erratic noise, dropouts, and cross-sensor inconsistencies.
- **Robust Pipeline**: Includes checkpointing, exponential backoff, rate-limit handling, and streaming writes to handle multi-gigabyte time-series datasets seamlessly.
- **Explainable Anomaly Attribution**: Provides interpretability for flagged anomalies to distinguish between genuine extreme weather events and faulty sensor hardware.

---

## 📂 Repository Structure

```text
SkyGuardAI/
├── .gitignore                   # Ignores large datasets (*.csv), caches, and checkpoints
├── requirements.txt             # Project dependencies
├── collect_aws_dataset.py       # High-res historical AWS data collector (Open-Meteo API)
├── eda.ipynb                    # Exploratory Data Analysis and data inspection notebook
├── plan.md                      # Comprehensive system design and architectural specification
└── README.md                    # Project documentation
```

---

## 🛠️ Getting Started

### 1. Prerequisites

Ensure you have Python 3.10 or higher installed.

```bash
git clone https://github.com/shyam1902005/SkyGuardAI.git
cd SkyGuardAI
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Collection

Run the high-res data collector to acquire historical data:

```bash
# Test with first 5 stations
python collect_aws_dataset.py --limit 5 --output sample_data.csv

# Collect all stations for custom date range
python collect_aws_dataset.py --start-date 2023-01-01 --end-date 2024-12-31
```

---

## 📄 License

This project is licensed under the MIT License.
