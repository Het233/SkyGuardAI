# 🤝 SkyGuard AI — Project Handover & Contributor Guide

Welcome to **SkyGuard AI**! This document provides everything you need to understand the codebase, set up your development environment, run the system, and start building new features or improvements.

---

## 📌 1. Project Overview & Mission

**SkyGuard AI** is an operational meteorological sensor surveillance and predictive maintenance platform engineered for India's **826 Automatic Weather Station (AWS)** network.

### The Strict 3-Variable Physical Constraint
To mirror real-world field conditions where AWS stations carry only primary sensors, SkyGuard AI operates **strictly on the primary three meteorological variables**:
1. **Temperature (°C)** (`temperature_c`)
2. **Surface Air Pressure (mbar / hPa)** (`air_pressure_mbar`)
3. **Relative Humidity (%)** (`relative_humidity_pct`)

*No secondary sensor variables (wind speed, solar radiation, rain gauges, radar) are required or used.* Spatial consensus and psychrometric thermodynamic consistency checks operate exclusively on this triad.

---

## 🏗️ 2. Architectural Pipeline

```text
               Raw 3-Variable Telemetry Stream (T, P, RH)
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Tiered Anomaly Detection                    │
  ├─────────────────────────────────────────────────────────────┤
  │ 1. Deterministic QC (Physical bounds & step-rate checks)    │
  │ 2. Rolling Statistical Baselines (Z-score, IQR)             │
  │ 3. Weather Isolation Forest (150 Unsupervised Trees)        │
  │ 4. Deep Autoencoder (PyTorch CUDA Manifold Reconstruction)  │
  │ 5. Temporal Residual Predictor (One-Step Forecasting Lags)  │
  │ 6. Multi-Station Spatial Consensus (Dynamic Lapse Rates)    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                Hybrid Anomaly Fusion Engine                 │
  │    (Deterministic Overrides, Model Consensus & Severity)    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
         ┌───────────────┐┌───────────────┐┌───────────────┐
         │Event-vs-Fault ││  Root-Cause   ││ Explainable   │
         │   Reasoning   ││  Classifier   ││   AI (XAI)    │
         │ (Heatwave vs  ││ (Random Forest││ (Diagnostic   │
         │ Sensor Fault) ││  12 Classes)  ││ Alert Cards)  │
         └───────┬───────┘└───────┬───────┘└───────┬───────┘
                 │                │                │
                 └───────────────┼────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
        ┌──────────────────┐            ┌──────────────────┐
        │  Sensor Health   │            │ Non-Destructive  │
        │     Scoring      │            │ Value Imputation │
        │  (0–100 Index &  │            │ (Spatial Lapse   │
        │  RUL Prediction) │            │ Reconstruction)  │
        └──────────────────┘            └──────────────────┘
```

---

## 📊 3. Current State & Benchmark Highlights

The models were trained on **26,762,400 observations** across all 826 Indian AWS stations from 1990–2025:

- **Ensemble F1-Score**: **99.74%**
- **Ensemble Recall**: **99.48%** (captures 99.5% of all atmospheric and sensor faults)
- **False Positive Rate**: **0.00%**
- **Inference Latency**: **< 1.0 µs** per observation (over 1M predictions/sec)
- **Regional Extreme Weather vs. Sensor Fault Discrimination**: **100.0%**
- **Fine-Grained Root Cause Classification**: Covers all **12 taxonomy classes**:
  `NORMAL`, `SPIKE`, `DROP`, `FROZEN_SENSOR`, `SENSOR_DRIFT`, `CONSTANT_OFFSET`, `HIGH_NOISE`, `MISSING_DATA`, `COMMUNICATION_CORRUPTION`, `PHYSICALLY_IMPOSSIBLE`, `CROSS_SENSOR_INCONSISTENCY`, `COORDINATED_MULTIVARIATE`.
- **Test Suite**: **51/51 automated unit tests passing**.

---

## ⚡ 4. Quickstart Guide (Local Setup)

### Step 1: Clone the Repo
```bash
git clone https://github.com/shyam1902005/SkyGuardAI.git
cd SkyGuardAI
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Extract Trained Models
The trained models are bundled in `skyguard_allindia_models.zip` (18.5 MB):
```bash
python -c "import zipfile; zipfile.ZipFile('skyguard_allindia_models.zip').extractall('.')"
```
This unpacks:
- `artifacts/models/autoencoder.pt`
- `artifacts/models/isolation_forest.pkl`
- `artifacts/models/temporal_predictor.pkl`
- `artifacts/models/root_cause_classifier.pkl`
- `artifacts/models/feature_engineer.pkl`
- `artifacts/models/training_metadata.json`

### Step 4: Verify with Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
# Output should show: Ran 51 tests ... OK
```

### Step 5: Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 6: (Optional) Launch the REST API
```bash
uvicorn api.server:app --reload --port 8000
```
Interactive Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📁 5. Directory Structure & Key Files

```text
SkyGuardAI/
├── anomaly_injection/        # Synthetic fault injection & physics taxonomy
│   ├── generator.py          # Master anomaly injection engine
│   └── taxonomy.py           # AnomalyType enum & physical bounds
├── api/                      # Production REST API (FastAPI)
│   ├── schemas.py            # Pydantic input/output schemas
│   └── server.py             # FastAPI streaming inference endpoints
├── artifacts/                # Trained models, benchmarks, and figures
│   ├── models/               # Model weights (.pt, .pkl)
│   └── *_allindia.*          # Master benchmark reports (Markdown & JSON)
├── baseline/                 # Phase 3 baseline detectors
│   ├── quality_control.py    # Deterministic physical limits & rate-of-change
│   └── statistical.py        # Rolling Z-score, IQR, and composite baseline
├── classification/           # Multiclass root-cause attribution
│   └── root_cause.py         # 12-class Random Forest & detector score attachment
├── dashboard/                # Operational Web UI (Streamlit)
│   └── app.py                # Live map, alert feed, diagnostics, sensor health
├── evaluation/               # Metric computation (Precision, Recall, F1, PR-AUC)
│   └── metrics.py            # Binary metrics & per-fault-type breakdown
├── explainability/           # Explainable AI (XAI)
│   ├── diagnostic_card.py    # Machine-readable fault diagnostic summary
│   └── explainer.py          # Feature attribution & physical violation explanations
├── features/                 # Meteorological feature engineering
│   └── engineering.py        # 50 features (lags, rolling stats, Fourier diurnal, psychrometric)
├── fusion/                   # Multi-tier decision fusion
│   └── engine.py             # Hybrid anomaly fusion (calibrated agreement & overrides)
├── health/                   # Sensor health index (0–100) & predictive maintenance
│   └── health_score.py       # Health index & Remaining Useful Life (RUL) estimation
├── imputation/               # Non-destructive self-healing imputation
│   └── correction.py         # Spatial lapse-rate reconstruction of anomalous readings
├── models/                   # Core machine learning & deep learning architectures
│   ├── autoencoder.py        # Deep Bottleneck Autoencoder (PyTorch CUDA)
│   ├── isolation_forest.py   # Weather Isolation Forest (150 trees)
│   ├── spatial_model.py      # Spatial Consensus Detector (K-NN lapse rates)
│   └── temporal_model.py     # Temporal Residual Predictor (One-step Ridge regression)
├── reasoning/                # Domain AI reasoning
│   └── event_vs_fault.py     # Distinguishes genuine extreme weather from sensor faults
├── tests/                    # Comprehensive unit tests (51 tests)
├── SkyGuard_AllIndia_Training.ipynb  # Full master Kaggle training pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project README
└── HANDOVER.md               # This contributor & handover guide
```

---

## 🎯 6. Roadmap of Recommended Future Improvements

Here are high-value areas where you can expand or improve SkyGuard AI:

### 1. Real-Time Streaming Ingestion (Kafka / MQTT)
- **Current State**: The system supports batch CSV processing and single/batch REST API requests.
- **Improvement**: Build a Kafka consumer or MQTT subscriber (`ingestion/stream_consumer.py`) that ingests live AWS telemetry packets in real time and routes them through the fusion engine.

### 2. Spatio-Temporal Graph Neural Network (ST-GNN)
- **Current State**: `SpatialConsensusDetector` uses K-Nearest Neighbors and barometric/thermal lapse rates.
- **Improvement**: Replace or augment K-NN with a Spatio-Temporal Graph Convolutional Network (e.g., PyTorch Geometric `GCNConv` or `A3T-GCN`) to model continuous, dynamic meteorological gradients between neighboring stations.

### 3. Automated Alert Webhooks & Notifications
- **Current State**: Diagnostic cards are displayed in the Streamlit UI and returned via the REST API.
- **Improvement**: Add an alert dispatcher (`explainability/alerts.py`) supporting webhooks for:
  - Telegram Bot API
  - Slack Incoming Webhooks
  - Email / SMS (Twilio or AWS SNS)
  Triggered automatically whenever an anomaly reaches `SeverityLevel.CRITICAL` or `HARDWARE_SENSOR_FAULT`.

### 4. Containerization & CI/CD
- **Current State**: Runs locally via virtual environment.
- **Improvement**:
  - Add a `Dockerfile` and `docker-compose.yml` for single-command deployment (FastAPI backend + Streamlit frontend).
  - Add a GitHub Actions workflow (`.github/workflows/tests.yml`) to automatically run the 51 unit tests on every pull request.

### 5. LLM-Powered Maintenance Work-Orders
- **Current State**: `diagnostic_card.py` outputs rule-based technical summaries.
- **Improvement**: Integrate a local small LLM (via Ollama, vLLM, or Gemini API) that translates diagnostic cards into natural-language work-orders with specific field repair steps (e.g. *"Inspect radiation shield ventilation; clean RTD sensor probe; recalibrate barometer offset"*).

---

## 🤝 Need Help?
- Check **`documentation.md`** for full deployment walkthroughs and FAQs.
- Check **`artifacts/ml_benchmark_allindia.md`** and **`artifacts/fusion_benchmark_allindia.md`** for detailed benchmark data.
- Run `python -m unittest discover tests` to verify any code changes you make.
