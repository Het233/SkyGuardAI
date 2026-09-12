# 🛰️ SkyGuard AI — Complete Operational Documentation

**Step-by-Step Guide to Running SkyGuard AI Locally on Your Laptop & Connecting to Cloudflare**

---

## 📋 Table of Contents
1. [System Overview & Core Philosophy](#1-system-overview--core-philosophy)
2. [Prerequisites & System Requirements](#2-prerequisites--system-requirements)
3. [Environment Setup & Installation](#3-environment-setup--installation)
4. [Running the Interactive Surveillance Dashboard (Streamlit)](#4-running-the-interactive-surveillance-dashboard-streamlit)
5. [Running the Real-Time Streaming REST API (FastAPI)](#5-running-the-real-time-streaming-rest-api-fastapi)
6. [One-Click Startup: Running Both Dashboard & API](#6-one-click-startup-running-both-dashboard--api)
7. [Running the End-to-End Pipeline Demo](#7-running-the-end-to-end-pipeline-demo)
8. [Running the Automated Test Suite (51 Unit Tests)](#8-running-the-automated-test-suite-51-unit-tests)
9. [Connecting to Your Custom Domain via Cloudflare Tunnel](#9-connecting-to-your-custom-domain-via-cloudflare-tunnel)
10. [Repository Directory Structure Cheatsheet](#10-repository-directory-structure-cheatsheet)
11. [Troubleshooting & FAQs](#11-troubleshooting--faqs)

---

## 1. System Overview & Core Philosophy

**SkyGuard AI** is an advanced meteorological sensor quality control, anomaly detection, predictive maintenance, and self-healing data imputation system designed for India's 826 Automatic Weather Station (AWS) network.

### The Strict 3-Variable Physical Constraint
To mirror realistic real-world conditions where weather stations only carry fundamental instruments, SkyGuard AI operates **strictly on the primary three meteorological variables**:
1. **Temperature (°C)** (`temperature_c`)
2. **Surface Air Pressure (mbar / hPa)** (`air_pressure_mbar`)
3. **Relative Humidity (%)** (`relative_humidity_pct`)

*No secondary sensor variables (wind speed, solar radiation, precipitation) are used.* Multi-station spatial consensus and psychrometric thermodynamic consistency checks operate exclusively on this triumvirate.

---

## 2. Prerequisites & System Requirements

Before running on your laptop, ensure you have:
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python:** Version `3.10` or higher (tested and verified on Python `3.11` and `3.14`).
- **Hardware:**
  - Minimum 4 GB RAM (8 GB+ recommended).
  - CPU: Any modern multi-core processor (Intel i5/i7/i9, AMD Ryzen, or Apple Silicon).
  - GPU *(Optional)*: NVIDIA GPU with CUDA support accelerates the PyTorch Deep Bottleneck Autoencoder; if absent, the model automatically runs on CPU.
- **Network / Ports:**
  - Port `8501` (Streamlit Dashboard)
  - Port `8000` (FastAPI REST Service)

---

## 3. Environment Setup & Installation

Follow these steps to configure your Python virtual environment on your laptop:

### Step 3.1: Clone or Navigate to the Repository
Open a terminal (PowerShell on Windows, or Terminal on macOS/Linux):
```powershell
cd E:\SkyGuardAI
```

### Step 3.2: Create a Virtual Environment
```powershell
python -m venv .venv
```

### Step 3.3: Activate the Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(If you get a script execution policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*
- **On Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **On Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 3.4: Install Required Packages
```powershell
pip install -r requirements.txt
```

---

## 4. Running the Interactive Surveillance Dashboard (Streamlit)

The Streamlit dashboard is the primary visual operations center for meteorologists and AWS station technicians.

### Launch Command:
```powershell
streamlit run dashboard/app.py
```
Or run directly with specific port and host settings:
```powershell
python -m streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

### Accessing the Dashboard:
Open your browser and navigate to:
```text
http://localhost:8501
```

### Overview of Dashboard Tabs:
1. **🗺️ Network Surveillance Map:**
   - Interactive map showing 10 Automatic Weather Stations across India (Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Ahmedabad, Pune, Srinagar, Guwahati).
   - Visual color-coded health markers (Green = Excellent, Cyan = Good, Amber = Degrading, Red = Critical).
2. **📈 Live Telemetry & Imputation Inspector:**
   - Dropdown selectors to pick any station and meteorological parameter.
   - Dual-trace Plotly visualization comparing:
     - **Corrupted Raw Observations** (Red markers)
     - **Pristine Target Values** (Translucent dotted gray)
     - **SkyGuard Non-Destructive Imputed Trajectory** (Emerald green solid curve)
   - Real-time Mean Absolute Error (MAE) and % error reduction metrics.
3. **🚨 XAI Diagnostic Alert Center:**
   - Live stream of diagnostic alert cards.
   - Breakdown of primary sensor culprit, confidence, step limit violations, and thermodynamic dew-point spread checks.
   - Standard Operating Procedure (SOP) maintenance work order (`IMD-AWS-STD-SOP-V4`) with targeted technician tasks.
4. **🩺 Prognostic Health & RUL Radar:**
   - Continuous 0–100 health score cards for all stations.
   - Breakdown of individual sensor scores: $H_T$, $H_P$, $H_{RH}$.
   - Remaining Useful Life (RUL) Days-to-Critical projections.
5. **⚡ Interactive Anomaly Injection Sandbox:**
   - An interactive testing lab where you can pick any station and inject on-the-fly faults:
     - `SPIKE` (impulse voltage jumps)
     - `DROP` (power supply dips)
     - `FROZEN_SENSOR` (ADC bus lockup)
     - `SENSOR_DRIFT` (calibration aging)
     - `HIGH_NOISE` (EMI interference)
     - `PHYSICALLY_IMPOSSIBLE` (envelope rupture)
   - Click **"🚀 Inject Anomaly & Run SkyGuard AI"** to see the system detect, classify root cause, generate a diagnostic card, and self-heal the reading in real time!

---

## 5. Running the Real-Time Streaming REST API (FastAPI)

For programmatic ingestion and automated data pipelines, SkyGuard AI provides an asynchronous REST service built with FastAPI.

### Launch Command:
```powershell
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive API Documentation:
Once started, open your browser:
- **Swagger Interactive UI:** [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **ReDoc UI:** [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

### Primary API Endpoints:
| Method | URL | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System liveness probe and device status (CPU/CUDA) |
| `POST` | `/api/v1/ingest/single` | Ingest single AWS observation, returns scores, XAI card & imputed values |
| `POST` | `/api/v1/ingest/batch` | Ingest multi-station synchronized observation frame |
| `GET` | `/api/v1/stations/health` | Get running 0–100 health scores & RUL for all stations |
| `GET` | `/api/v1/stations/{id}/health` | Get detailed health breakdown for a specific station |
| `GET` | `/api/v1/alerts/active` | Retrieve recent unacknowledged diagnostic alerts |

### Example cURL Request (Single Observation Ingestion):
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/single" \
     -H "Content-Type: application/json" \
     -d '{
       "station_id": "IMD_AWS_0001",
       "timestamp": "2024-01-01T12:00:00",
       "temperature_c": 32.4,
       "air_pressure_mbar": 1008.2,
       "relative_humidity_pct": 55.0,
       "latitude": 28.584,
       "longitude": 77.206,
       "elevation_m": 216.0
     }'
```

---

## 6. One-Click Startup: Running Both Dashboard & API

To start both the FastAPI backend and Streamlit dashboard together in the background with a single command, run the included launcher script:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_services.ps1
```

Or start them selectively:
```powershell
# Start only the Streamlit dashboard:
powershell -ExecutionPolicy Bypass -File .\start_services.ps1 -Service dashboard

# Start only the FastAPI backend:
powershell -ExecutionPolicy Bypass -File .\start_services.ps1 -Service api
```

---

## 7. Running the End-to-End Pipeline Demo

To run an end-to-end batch evaluation across 15,000 observations (multi-tier scoring, event-vs-fault reasoning, root-cause classification, XAI card generation, health tracking, and value imputation):

```powershell
python run_pipeline_demo.py --input data/synthetic/sample_synthetic_anomalies.csv --limit 15000
```

### Outputs Generated:
- `artifacts/pipeline_demo_report.md` (Markdown executive summary)
- `artifacts/pipeline_demo_results.json` (Structured JSON benchmark data)

---

## 8. Running the Automated Test Suite (51 Unit Tests)

To verify the integrity of all models, baseline checks, detectors, API endpoints, and imputation modules:

```powershell
python -m unittest discover tests -v
```

**Expected output:**
```text
Ran 51 tests in ~6.5s
OK
```

---

## 9. Connecting to Your Custom Domain via Cloudflare Tunnel

Your PC already has `cloudflared` installed and running as an active background Windows service under the tunnel name **`my-laptop-tunnel`**. 

Because this is a **remote-managed Zero Trust tunnel**, you do not need to configure anything on your router. Follow these steps to map your custom domain:

### Step 9.1: Open Cloudflare Zero Trust Dashboard
1. Open your browser and navigate to **[one.dash.cloudflare.com](https://one.dash.cloudflare.com/)**.
2. Log in with your Cloudflare account.

### Step 9.2: Navigate to Tunnels
1. In the left sidebar, click **Networks** ➔ **Tunnels** (or **Access** ➔ **Tunnels**).
2. Locate the existing tunnel: **`my-laptop-tunnel`** (Status should be **HEALTHY** / Green).
3. Click the three dots `...` next to `my-laptop-tunnel` and select **Configure**.

### Step 9.3: Add Public Hostname (Map Your Domain)
1. Go to the **Public Hostname** tab.
2. Click the blue **Add a public hostname** button.
3. Fill in the fields:
   - **Subdomain:** `skyguard` *(e.g., `skyguard.yourdomain.com`)* or leave blank for root domain (`@`).
   - **Domain:** Select your registered domain from the dropdown.
   - **Path:** Leave blank.
   - **Type:** Select **`HTTP`**.
   - **URL:** Type **`localhost:8501`** *(routes to Streamlit Dashboard)*.
4. Click **Save hostname** at the bottom right.

*(Optional: To also expose the REST API on a separate subdomain, click **Add a public hostname** again, enter Subdomain: `api-skyguard`, Type: `HTTP`, URL: `localhost:8000`)*.

### Step 9.4: Access Remotely!
You can now visit your domain from any computer, tablet, or phone worldwide:
```text
https://skyguard.yourdomain.com
```
- Fully encrypted HTTPS with free SSL certificates automatically managed by Cloudflare.
- Zero open ports or port-forwarding required on your local Wi-Fi router.
- Pre-configured `.streamlit/config.toml` ensures WebSocket connections work without CORS errors.

---

## 10. Repository Directory Structure Cheatsheet

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
├── start_services.ps1           # One-click launcher for Dashboard & API
├── run_pipeline_demo.py         # End-to-end integration demo script
├── documentation.md             # This comprehensive setup & user guide
├── requirements.txt             # Project dependencies
└── README.md                    # Project overview & scientific benchmarks
```

---

## 11. Troubleshooting & FAQs

### Q1: "Running scripts is disabled on this system" error in PowerShell
**Fix:** Run this command once in your PowerShell terminal to allow local scripts to execute:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Q2: Port 8501 or 8000 is already in use
**Fix:** To check which process is occupying the port and free it:
```powershell
# Check what is using port 8501:
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue

# Stop existing Python processes if needed:
Stop-Process -Name python -Force
```

### Q3: Streamlit shows "Please replace the URL with localhost..." when accessing through Cloudflare
**Fix:** This is caused by Streamlit's default CORS/XSRF protection. We have already created `.streamlit/config.toml` with `enableCORS = false` and `enableXsrfProtection = false` to resolve this automatically.

### Q4: How do I verify my Cloudflare Tunnel is connected?
**Fix:** Run in PowerShell:
```powershell
cloudflared tunnel info my-laptop-tunnel
```
If it shows active connections to edge servers (e.g. `1xbom03`), your tunnel is live and healthy.
