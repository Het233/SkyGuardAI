# 🛰️ SkyGuard AI

**Intelligent Meteorological Sensor Quality Control, Multi-Tier Anomaly Detection, Predictive Maintenance & Self-Healing Imputation for Automatic Weather Station (AWS) Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%201.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Unit Tests](https://img.shields.io/badge/Tests-51%20Passed-success.svg)](tests/)

---

## 📌 Mission & Overview

**SkyGuard AI** is an operational, production-grade meteorological sensor surveillance system engineered to detect real-time anomalies, sensor degradation, and communication failures across India's 826 Automatic Weather Station (AWS) network.

### The Strict 3-Variable Physical Constraint
To preserve physical realism and operational viability across varied field networks, SkyGuard AI operates strictly on the primary triumvirate of core atmospheric variables:
1. **Temperature (°C)** (`temperature_c`)
2. **Surface Air Pressure (mbar / hPa)** (`air_pressure_mbar`)
3. **Relative Humidity (%)** (`relative_humidity_pct`)

*No supplementary parameters (e.g., wind speed, solar irradiance, rainfall gauges, radar) are required or utilized.* Spatial neighbor consensus and thermodynamic coupling checks operate exclusively across this identical trio.

---

## 🏛️ System Architecture

```text
                     Raw 3-Variable Ingest Stream
          (Temperature, Surface Pressure, Relative Humidity)
                                  │
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │       Multi-Tier Anomaly Detection Tiers          │
        ├───────────────────────────────────────────────────┤
        │ 1. Deterministic QC (Physical limits & step rate) │
        │ 2. Rolling Statistical Baselines (Z-score, IQR)   │
        │ 3. Weather Isolation Forest (150 Trees)           │
        │ 4. Deep Bottleneck Autoencoder (PyTorch CUDA)     │
        │ 5. Temporal Trajectory Regressor (Diurnal Lags)   │
        │ 6. Multi-Station Spatial Consensus (Lapse Rates)  │
        └─────────────────────────┬─────────────────────────┘
                                  │
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │           Hybrid Anomaly Fusion Engine            │
        │  (Deterministic Override, Agreement & Severity)   │
        └─────────────────────────┬─────────────────────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
         ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
         │ Event-vs-Fault│ │   Root-Cause  │ │  Explainable  │
         │   Reasoning   │ │   Classifier  │ │   AI (XAI)    │
         │ (Heatwave vs  │ │ (Random Forest│ │ (Diagnostic   │
         │ Sensor Fault) │ │  12 Classes)  │ │ Alert Cards)  │
         └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                 │                 │                 │
                 └────────────────┼────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
        ┌──────────────────┐             ┌──────────────────┐
        │  Sensor Health   │             │ Non-Destructive  │
        │     Scoring      │             │ Value Imputation │
        │  (0–100 Index &  │             │ (Zero Overwrite  │
        │  RUL Prediction) │             │  Spatial Lapse)  │
        └──────────────────┘             └──────────────────┘
```

---

## 📊 Benchmark Performance Results

Evaluated on **87,600 observations** across 10 Automatic Weather Stations (4,471 ground-truth anomalies, 5.10% anomaly rate):

| Architecture Tier | Precision | Recall | F1-Score | PR-AUC | False Positive Rate | Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic QC (Phase 3)** | **0.8118** | 0.4603 | 0.5875 | 0.4208 | **0.57%** | 3.0 µs |
| **Statistical Baseline (Phase 3)** | 0.6692 | 0.5833 | 0.6233 | 0.5144 | 1.55% | 9.6 µs |
| **Isolation Forest (Phase 4)** | 0.2173 | 0.2948 | 0.2502 | 0.1740 | 5.71% | 3.7 µs |
| **Deep Autoencoder (CUDA) (Phase 4)** | 0.2439 | 0.7886 | 0.3726 | 0.2660 | 13.15% | 2.1 µs |
| **Temporal Residual (Phase 4)** | 0.2843 | 0.0946 | 0.1420 | 0.1901 | 1.28% | **0.2 µs** |
| **Spatial Consensus (Phase 5)** | 0.1295 | 0.7656 | 0.2216 | 0.1553 | 27.67% | 5.4 µs |
| **SkyGuard Complete Hybrid Engine** | **0.5413** | **0.8202** | **0.6521** | **0.6614** | **3.74%** | **0.1 µs** |

### Fault Taxonomy Detection Breakdown
- **100.0%**: `FROZEN_SENSOR`, `DROP`, `PHYSICALLY_IMPOSSIBLE`, `MISSING_DATA`, `COORDINATED_MULTIVARIATE`, `CROSS_SENSOR_INCONSISTENCY`, `COMMUNICATION_CORRUPTION`
- **97.8%**: `CONSTANT_OFFSET` (791 / 809)
- **92.6%**: `SPIKE` (50 / 54)
- **89.6%**: `HIGH_NOISE` (414 / 462)
- **61.9%**: `SENSOR_DRIFT` (1,193 / 1,927) *(Doubled from 31.0% in Phase 3)*

### Multiclass Diagnosis & Imputation Accuracy
- **Root-Cause Multiclass Accuracy:** **92.62%** (0.8993 Macro-F1 across all 12 fault types)
- **Event-vs-Fault Discrimination:** **100.0%** attribution accuracy on regional synchronous heatwaves
- **Self-Healing Imputation Fidelity:**
  - **Surface Pressure:** Raw MAE 307.37 mbar $\rightarrow$ Imputed MAE **7.92 mbar** (**97.4% error reduction**)
  - **Temperature:** Raw MAE 2.98°C $\rightarrow$ Imputed MAE **0.91°C** (**69.6% error reduction**)
  - **Relative Humidity:** Raw MAE 8.19% $\rightarrow$ Imputed MAE **5.82%** (**29.0% error reduction**)

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/shyam1902005/SkyGuardAI.git
cd SkyGuardAI

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python -m unittest discover tests
# Ran 51 tests in 6.46s -> OK
```

### 3. Run End-to-End Pipeline Demonstration
```bash
python run_pipeline_demo.py --input data/synthetic/sample_synthetic_anomalies.csv --limit 15000
```

### 4. Launch Streaming REST API Server
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be live at `http://localhost:8000/docs`.

### 5. Launch Meteorological Surveillance Dashboard
```bash
streamlit run dashboard/app.py
```
Opens the command-center surveillance interface at `http://localhost:8501`.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness probe and runtime system diagnostics |
| `POST` | `/api/v1/ingest/single` | Ingest single AWS observation, return scores, XAI card & imputed values |
| `POST` | `/api/v1/ingest/batch` | Ingest multi-station synchronized observation frame |
| `GET` | `/api/v1/stations/health` | Retrieve running 0–100 health scores & RUL for all stations |
| `GET` | `/api/v1/stations/{id}/health` | Retrieve detailed health breakdown for a specific AWS station |
| `GET` | `/api/v1/alerts/active` | Query active unacknowledged diagnostic alert cards |

---

## 📂 Repository Structure

```text
SkyGuardAI/
├── api/                         # FastAPI Streaming REST Service (Phase 11)
│   ├── schemas.py               # Pydantic input/output schemas
│   └── server.py                # REST endpoints & model buffer manager
├── anomaly_injection/           # Synthetic Fault Taxonomy Generator (Phase 2)
│   ├── generator.py             # Master injection engine
│   └── taxonomy.py              # 11 fault classes & ground-truth events
├── baseline/                    # Deterministic QC & Statistical Baselines (Phase 3)
│   ├── quality_control.py       # Physical bounds, step limits, persistence
│   └── statistical.py           # Z-Score, IQR, EWMA residuals
├── classification/              # Multiclass Root-Cause Classifier (Phase 7)
│   └── root_cause.py            # Supervised 100-tree Random Forest
├── dashboard/                   # Interactive Surveillance Dashboard (Phase 12)
│   └── app.py                   # Streamlit operations command center
├── explainability/              # Explainable AI (XAI) Engine (Phase 8)
│   ├── explainer.py             # Sensor culprit & feature attribution
│   └── diagnostic_card.py       # Standardized IMD maintenance alert cards
├── features/                    # Meteorological Feature Engineering (Phase 4)
│   └── engineering.py           # 50 diurnal, psychrometric & lag features
├── fusion/                      # Hybrid Anomaly Fusion Engine (Phase 5)
│   └── engine.py                # Calibrated 6-tier composite scoring
├── health/                      # Sensor Health Scoring & Prognostics (Phase 9)
│   └── health_score.py          # 0-100 health indices & Days-to-Critical RUL
├── imputation/                  # Non-Destructive Value Imputation (Phase 10)
│   └── correction.py            # Spatial lapse & diurnal self-healing
├── models/                      # ML & Deep Learning Detectors (Phases 4 & 5)
│   ├── autoencoder.py           # Deep Bottleneck Manifold Model (PyTorch CUDA)
│   ├── isolation_forest.py      # Weather Isolation Forest
│   ├── spatial_model.py         # Multi-station spatial consensus & lapse rates
│   └── temporal_model.py        # Temporal diurnal residual predictor
├── reasoning/                   # Event-vs-Fault Physical Reasoning (Phase 6)
│   └── event_vs_fault.py        # Multi-station synchronicity & thermodynamics
├── tests/                       # Automated Test Suite (51 Unit Tests)
├── run_pipeline_demo.py         # End-to-end integration demo script
├── requirements.txt             # Environment dependencies
└── README.md                    # Project documentation
```

---

## 📄 License
This project is licensed under the MIT License.
