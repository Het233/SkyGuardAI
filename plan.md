Yes. The project is feasible, but there is one important constraint to resolve up front:

> The core detector can use **only Temperature, Pressure, and Humidity as sensor variables**.
> For the example involving “neighboring stations,” spatial comparison requires **the same three variables from nearby AWS stations**. That is still consistent with the constraint if no additional meteorological variables are introduced.

## SkyGuard AI — Complete Implementation Plan

### 1. System architecture

```text
AWS Sensors
   │
   ├── Temperature
   ├── Pressure
   └── Humidity
   │
   ▼
Data Ingestion Layer
   │
   ▼
Preprocessing + Validation
   │
   ├── Missing-value checks
   ├── Range checks
   ├── Duplicate/timestamp checks
   ├── Rate-of-change checks
   └── Unit/format validation
   │
   ▼
Feature Engineering
   │
   ├── Temporal features
   ├── Rolling statistics
   ├── Cross-sensor relationships
   ├── Station-to-station consistency
   └── Seasonal/time-of-day features
   │
   ▼
AI Anomaly Engine
   │
   ├── Statistical detector
   ├── Isolation Forest
   ├── Autoencoder
   ├── Temporal prediction/residual model
   └── Multivariate consistency model
   │
   ▼
Anomaly Fusion Layer
   │
   ├── Anomaly probability
   ├── Severity
   ├── Confidence
   └── Root-cause classification
   │
   ▼
Explainability Layer
   │
   └── SHAP / feature contribution
   │
   ▼
Sensor Health + Prediction
   │
   ├── Sensor health score
   ├── Degradation trend
   └── Maintenance recommendation
   │
   ▼
Dashboard / API / Alerts
```

---

# 2. Data strategy

Use **two data sources**.

### A. Real historical AWS data

Use historical observations containing:

```text
timestamp
station_id
temperature
pressure
humidity
```

Preferably multiple stations and multiple months/years.

We should preserve timestamps because temporal behavior is central to the project.

### B. Synthetic anomaly injection

Because genuinely labeled faulty AWS observations are difficult to obtain, construct a clean dataset and inject realistic faults.

Inject at least:

| Anomaly                    | Example                                                              |
| -------------------------- | -------------------------------------------------------------------- |
| Spike                      | 25 → 55 → 25 °C                                                      |
| Drop                       | 28 → 5 → 28 °C                                                       |
| Frozen sensor              | 28.4, 28.4, 28.4, 28.4...                                            |
| Constant offset            | actual + 8 °C                                                        |
| Gradual drift              | +0.05 °C/hour                                                        |
| Noise                      | unusually high random fluctuation                                    |
| Missing data               | NaN / absent packets                                                 |
| Communication corruption   | duplicated/out-of-order packets                                      |
| Impossible values          | RH > 100%, negative pressure                                         |
| Cross-sensor inconsistency | temperature changes but humidity/pressure do not behave consistently |
| Coordinated anomaly        | multiple variables suddenly become unrealistic                       |
| Seasonal deviation         | observation inconsistent with expected seasonal/time-of-day behavior |

This gives us controllable ground truth for evaluation.

---

# 3. Preprocessing layer

Before AI, implement deterministic quality control.

### Basic validation

```text
Temperature:
    physically/plausibly valid range

Pressure:
    plausible atmospheric range

Humidity:
    0–100 %

Timestamp:
    valid
    ordered
    no unexpected duplication
```

Do **not** rely on these rules as the final detector.

They are only the first defensive layer.

---

# 4. Feature engineering

This is one of the most important parts.

For each observation, generate:

### Temporal features

```text
current value
lag-1
lag-5
lag-15
lag-30
rolling mean
rolling std
rolling min/max
rate of change
acceleration/change-of-change
```

Example:

```text
temperature_rate =
T(t) - T(t-1)
```

### Time/season features

```text
hour
day of year
month
season
sin(hour)
cos(hour)
sin(day_of_year)
cos(day_of_year)
```

This allows the model to distinguish:

```text
35 °C at 2 PM in summer
```

from

```text
35 °C at 3 AM in winter
```

---

# 5. Multivariate consistency

The three parameters should not be treated as completely independent.

Create relationships such as:

```text
Temperature ↔ Humidity
Temperature ↔ Pressure
Humidity ↔ Temperature variation
Pressure temporal behavior
```

Instead of assuming a fixed physical equation, learn the **normal joint distribution** from historical data.

This is important because an anomaly can look normal individually but abnormal collectively.

Example:

```text
Temperature = 55 °C
Humidity = 98 %
Pressure = 870 hPa
```

Each value might pass some simplistic checks, but their joint pattern can be highly improbable.

---

# 6. Spatial consistency

For the stronger version of SkyGuard, use:

```text
Station A
Station B
Station C
Station D
```

and compare the same three variables.

For example:

```text
A = 55 °C
B = 29 °C
C = 28 °C
D = 30 °C
```

The system can recognize that A is an outlier relative to its neighboring stations.

### Important constraint

Do **not** use:

```text
wind
rainfall
solar radiation
visibility
etc.
```

unless the competition explicitly allows additional variables.

The spatial module still operates using only:

```text
temperature
pressure
humidity
```

---

# 7. AI anomaly detection strategy

I would **not** build this around a single model.

Use an ensemble/hybrid architecture.

### Detector 1 — Statistical baseline

Examples:

```text
Z-score
IQR
EWMA
rolling statistics
```

Purpose:

Fast detection of obvious anomalies.

---

### Detector 2 — Isolation Forest

Input:

```text
T
P
RH
temporal features
rolling statistics
```

Good for detecting unusual observations without requiring anomaly labels.

---

### Detector 3 — Autoencoder

Train an autoencoder on **normal observations only**.

```text
Input:
[T, P, RH, temporal features]

        ↓

Encoder

        ↓

Latent representation

        ↓

Decoder

        ↓

Reconstructed values
```

Calculate:

```text
reconstruction_error
```

Large reconstruction error → potential anomaly.

---

### Detector 4 — Temporal prediction model

A lightweight temporal model can predict:

```text
T(t+1)
P(t+1)
RH(t+1)
```

using recent observations.

Possible model:

```text
LSTM
GRU
Temporal CNN
small Transformer
```

For deployment on ESP32, use a much smaller model.

Then calculate:

```text
prediction_residual =
actual - predicted
```

Large residual → suspicious observation.

---

### Detector 5 — Cross-station detector

For multi-station deployment:

```text
station value
      vs
neighboring station distribution
```

Possible techniques:

```text
robust z-score
distance-based anomaly score
small neural model
Isolation Forest
```

---

# 8. Final anomaly score

Combine the detectors instead of trusting one model.

For example:

```text
Final Score =
w1 × statistical_score
+ w2 × isolation_score
+ w3 × autoencoder_score
+ w4 × temporal_score
+ w5 × spatial_score
```

Normalize everything to:

```text
0 → 1
```

Then classify:

```text
0.00–0.30   Normal
0.30–0.60   Suspicious
0.60–0.80   Anomaly
0.80–1.00   Critical
```

The exact thresholds should be determined from validation data rather than arbitrarily claimed as optimal.

---

# 9. Genuine event vs sensor anomaly

This is a **core research contribution**.

A simple detector says:

> “This value is unusual.”

SkyGuard should attempt to answer:

> “Is this unusual value likely to be a real atmospheric event or a faulty observation?”

### Logic

If multiple nearby stations simultaneously show:

```text
temperature increase
humidity decrease
pressure change
```

then the event is more likely genuine.

If:

```text
only one station changes dramatically
neighboring stations remain stable
```

then sensor fault probability increases.

So generate two probabilities:

```text
P(genuine_event)
P(sensor_anomaly)
```

---

# 10. Root-cause classification

After detecting an anomaly, classify it.

Possible classes:

```text
NORMAL
SPIKE
DROP
FROZEN_SENSOR
SENSOR_DRIFT
NOISE
MISSING_DATA
DUPLICATE_DATA
COMMUNICATION_ERROR
CROSS_SENSOR_INCONSISTENCY
SPATIAL_INCONSISTENCY
POSSIBLE_CALIBRATION_ERROR
UNKNOWN
```

This can initially be done using an ML classifier trained on the synthetic anomaly labels.

---

# 11. Explainable AI

For every alert, the system should provide something like:

```text
Anomaly detected

Station: AWS_014
Timestamp: 14:32:10

Anomaly Score: 0.94
Confidence: 0.91
Severity: CRITICAL

Likely Cause:
Temperature sensor spike

Evidence:
Temperature contribution: +0.61
Humidity inconsistency: +0.18
Temporal deviation: +0.12
Spatial deviation: +0.09
```

Use:

```text
SHAP
```

primarily.

LIME can be included as an alternative, but I would make SHAP the main explainability mechanism.

---

# 12. Sensor health score

Instead of only detecting individual bad readings, maintain a long-term sensor health score.

Example:

```text
Sensor Health = 87/100
```

Factors:

```text
anomaly frequency
drift magnitude
missing-data rate
communication failures
reconstruction error trend
prediction residual trend
```

Example:

```text
Last 24h anomalies: 12
Last 7d anomalies: 47
Estimated drift: +0.08 °C/day

Status:
WARNING

Recommendation:
Inspect/recalibrate temperature sensor.
```

---

# 13. Predictive maintenance

Once the health score exists, estimate degradation.

A simple first implementation:

```text
health(t)
```

over time.

Then detect:

```text
consistent downward trend
```

Possible future extension:

```text
regression / survival model
```

to estimate:

```text
maintenance risk in next N days
```

I would **not overclaim actual remaining useful life** unless the dataset contains sufficiently rich degradation histories.

For the competition, call it:

> **maintenance-risk prediction**

rather than pretending we can precisely predict failure date.

---

# 14. Corrected/imputed value

For anomalous readings, generate:

```text
Observed:
Temperature = 55 °C

Estimated:
Temperature = 29.4 °C
```

Methods can include:

```text
temporal model prediction
rolling interpolation
autoencoder reconstruction
neighbor-station consensus
```

Show both:

```text
Observed value
Estimated normal value
```

Never silently overwrite the raw data.

---

# 15. Real-time pipeline

The deployed pipeline should look like:

```text
Sensor
  ↓
MQTT / HTTP
  ↓
Stream processor
  ↓
Feature generation
  ↓
Anomaly inference
  ↓
Alert engine
  ↓
Database
  ↓
Dashboard
```

A practical prototype stack:

```text
Python
FastAPI
MQTT
Kafka or lightweight queue
Pandas / NumPy
Scikit-learn
PyTorch
SHAP
SQLite/PostgreSQL
Streamlit
```

For a competition prototype, **Kafka is optional**. Using it just because it sounds scalable is unnecessary.

Start with:

```text
FastAPI + MQTT + Python
```

and design the inference service so it can later be horizontally scaled.

---

# 16. Dashboard

Build a dashboard with:

### Live station view

```text
Station
Temperature
Pressure
Humidity
Status
Health
Latest anomaly
```

### Time-series graph

```text
Temperature over time
Pressure over time
Humidity over time
```

Anomalous points are highlighted.

### Alert panel

```text
TIME
STATION
TYPE
SEVERITY
CONFIDENCE
STATUS
```

### Explainability panel

Show:

```text
Why was this observation flagged?
```

### Sensor health panel

```text
Healthy
Warning
Degrading
Critical
```

### Optional map

Plot stations geographically **only if station coordinates are available and permitted**. Coordinates are metadata, not an additional meteorological sensor parameter.

---

# 17. Edge AI / ESP32

Do **not** try to deploy the complete SkyGuard system onto ESP32.

Instead create:

```text
Cloud/Server Model
+
Tiny Edge Model
```

### ESP32 handles

```text
range validation
rate-of-change check
small statistical detector
small quantized ML model
```

### Server handles

```text
ensemble detection
spatial analysis
SHAP
root-cause analysis
maintenance prediction
dashboard
```

Potential edge model:

```text
tiny neural network
or
compact decision tree
or
quantized autoencoder
```

Use:

```text
TensorFlow Lite Micro
```

only after the main system works.

---

# 18. Evaluation methodology

This is critical because your evaluation explicitly mentions injected anomalies.

Create:

```text
Clean dataset
      ↓
Synthetic anomaly generator
      ↓
Ground-truth labels
      ↓
SkyGuard
      ↓
Predictions
```

Measure:

### Detection

```text
Precision
Recall
F1-score
ROC-AUC
PR-AUC
```

For highly imbalanced anomaly data, **PR-AUC and F1** are particularly important.

### False alarms

Measure:

```text
False Positive Rate
False Alerts / hour
False Alerts / station / day
```

because minimizing false alarms is explicitly part of the problem.

### Real-time

Measure:

```text
average latency
p95 latency
throughput
CPU usage
memory
```

### Edge

Measure:

```text
RAM
flash
inference time
energy/inference
```

### Explainability

Conduct a small human evaluation:

```text
Can a user understand why the system flagged this value?
```

---

# 19. Benchmarking

To make the research stronger, compare SkyGuard against:

```text
Threshold rules
Z-score
Isolation Forest
Autoencoder
Temporal model
SkyGuard ensemble
```

Example table:

| Method           | Precision | Recall | F1 | Latency |
| ---------------- | --------: | -----: | -: | ------: |
| Threshold        |         — |      — |  — |       — |
| Z-score          |         — |      — |  — |       — |
| Isolation Forest |         — |      — |  — |       — |
| Autoencoder      |         — |      — |  — |       — |
| LSTM             |         — |      — |  — |       — |
| **SkyGuard**     |         — |      — |  — |       — |

We should **fill these only from actual experiments**, not invent them.

---

# 20. Recommended project phases

### Phase 1 — Dataset

```text
Obtain AWS data
Clean it
Normalize schema
Create train/validation/test sets
```

### Phase 2 — Anomaly generator

Implement reproducible injection:

```python
inject_spike()
inject_drift()
inject_freeze()
inject_missing()
inject_noise()
inject_communication_error()
...
```

This gives us controlled ground truth.

### Phase 3 — Baseline system

Implement:

```text
range rules
Z-score
IQR
rolling statistics
```

### Phase 4 — ML detector

Implement:

```text
Isolation Forest
Autoencoder
Temporal model
```

### Phase 5 — Hybrid fusion

Implement:

```text
multi-model anomaly score
```

### Phase 6 — Event-vs-fault reasoning

Add:

```text
temporal consistency
multivariate consistency
spatial consistency
```

### Phase 7 — Root cause

Train classifier for:

```text
spike
drift
freeze
communication error
etc.
```

### Phase 8 — Explainability

Add:

```text
SHAP
```

### Phase 9 — Sensor health

Implement:

```text
health score
degradation trend
maintenance risk
```

### Phase 10 — Imputation

Generate:

```text
estimated corrected value
```

### Phase 11 — API + streaming

Create:

```text
FastAPI
MQTT ingestion
real-time inference
```

### Phase 12 — Dashboard

Build the complete UI.

### Phase 13 — ESP32

Compress/quantize the lightweight detector.

### Phase 14 — Evaluation

Run all injected anomaly experiments and generate:

```text
metrics
plots
confusion matrices
latency measurements
ablation study
```

---

# 21. Important research experiment: ablation study

This can significantly strengthen the paper.

Compare:

```text
A: temporal only
B: multivariate only
C: temporal + multivariate
D: temporal + multivariate + spatial
E: complete SkyGuard
```

Then demonstrate exactly what each component contributes.

For example:

```text
Spatial consistency reduces false positives.
Temporal prediction improves drift detection.
Multivariate consistency detects physically inconsistent combinations.
```

Again, those conclusions should come from our measurements rather than being assumed beforehand.

---

# 22. Final executable repository structure

I recommend building the repository approximately like this:

```text
skyguard-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── configs/
│   └── config.yaml
│
├── ingestion/
│   ├── mqtt_client.py
│   └── stream_processor.py
│
├── preprocessing/
│   ├── validation.py
│   ├── cleaning.py
│   └── features.py
│
├── anomaly_injection/
│   ├── spikes.py
│   ├── drift.py
│   ├── freeze.py
│   ├── missing.py
│   └── generator.py
│
├── models/
│   ├── isolation_forest.py
│   ├── autoencoder.py
│   ├── temporal_model.py
│   ├── spatial_model.py
│   └── fusion.py
│
├── classification/
│   └── root_cause.py
│
├── explainability/
│   └── shap_explainer.py
│
├── health/
│   ├── health_score.py
│   └── maintenance.py
│
├── imputation/
│   └── correction.py
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── edge/
│   └── esp32/
│
├── evaluation/
│   ├── metrics.py
│   ├── benchmark.py
│   └── plots.py
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
├── Dockerfile
├── README.md
└── run_demo.py
```

---

# 23. Example final output

For your example:

```text
Station: AWS_021
Time: 14:32:10

Temperature: 55.0 °C
Pressure: 870 hPa
Humidity: 98 %

--------------------------------
ANOMALY DETECTED
--------------------------------

Anomaly Score: 0.96
Confidence: 0.93
Severity: CRITICAL

Root Cause:
Probable temperature sensor malfunction

Evidence:
✓ Extreme temporal deviation
✓ Strong multivariate inconsistency
✓ Strong spatial disagreement
✓ High reconstruction error
✓ High prediction residual

Event Probability: 0.08
Sensor Fault Probability: 0.92

Estimated Normal Temperature:
29.4 °C

Sensor Health:
62 / 100

Recommendation:
Inspect temperature sensor and calibration.
```

That is the kind of end-to-end behavior the final demo should visibly demonstrate.

---

# 24. What the final deliverables should contain

### Deliverable 1 — Fully executable code

It should support:

```text
dataset → preprocessing → training → anomaly injection
→ inference → root cause → explanation → health → dashboard
```

with a single documented demo command.

### Deliverable 2 — Example dataset

Include a small dataset so the project runs **without requiring the evaluator to download anything**.

### Deliverable 3 — Synthetic anomaly generator

This is essential for reproducing the evaluation.

### Deliverable 4 — Trained model artifacts

Provide:

```text
model files
scalers
encoders
configuration
```

where licensing permits.

### Deliverable 5 — Dashboard

A working real-time or simulated-stream interface.

### Deliverable 6 — Documentation

The document should explain:

```text
1. Problem
2. System architecture
3. Dataset
4. Preprocessing
5. Feature engineering
6. Anomaly types
7. Models
8. Ensemble scoring
9. Event-vs-fault reasoning
10. Root-cause classification
11. SHAP explainability
12. Sensor health
13. Maintenance prediction
14. Imputation
15. Real-time deployment
16. ESP32 architecture
17. Evaluation methodology
18. Results
19. Limitations
20. Future work
21. Use cases
22. Installation
23. Example execution
```

### Deliverable 7 — Research material

Generate:

```text
architecture diagram
workflow diagram
model comparison
confusion matrix
ROC/PR curves
latency graphs
anomaly examples
ablation results
```

---

## One design decision I strongly recommend

Don't pitch this as merely:

> **“We use Isolation Forest to detect weather anomalies.”**

That is relatively ordinary.

The stronger research proposition is:

> **SkyGuard is a hierarchical, explainable, real-time anomaly detection framework that combines temporal behavior, multivariate atmospheric consistency, optional spatial consensus, anomaly-type classification, and sensor-health estimation to distinguish genuine meteorological events from faulty AWS observations.**

That positioning directly addresses **accuracy + false alarms + explainability + scalability + deployability**, which aligns much better with the evaluation criteria.
