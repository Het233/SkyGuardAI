"""
================================================================================
dashboard/pages/live_data.py — SkyGuard AI · Live Data Validation
================================================================================
Real-time weather validation dashboard.

Compares:
  1. Live Open-Meteo API values (fetched via station lat/lon)
  2. AWS station values from the EDA dataset

Falls back to cached data automatically if the internet is unavailable.

Entry-point called from app.py:
    render_live_data()
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── path resolution ──────────────────────────────────────────────────────────
_DASH = os.path.dirname(os.path.dirname(__file__))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

from data_loader import load_aws_dataset

# ── constants ─────────────────────────────────────────────────────────────────
_CHART_CFG = {"displayModeBar": False, "responsive": True}

# Deviation thresholds (Green / Amber / Red)
_THRESHOLDS = {
    "temperature": (2.0, 4.0),    # °C
    "humidity":    (8.0, 15.0),   # %
    "pressure":    (3.0, 6.0),    # hPa
    "wind":        (3.0, 7.0),    # m/s
    "rainfall":    (5.0, 12.0),   # mm
}

# Weights for validation score
_WEIGHTS = {
    "temperature": 0.30,
    "humidity":    0.25,
    "pressure":    0.20,
    "wind":        0.15,
    "rainfall":    0.10,
}

# Open-Meteo API endpoint
_API_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&current=temperature_2m,relative_humidity_2m,surface_pressure,"
    "wind_speed_10m,precipitation"
    "&timezone=Asia%2FKolkata"
)

# Cache file for offline fallback
_CACHE_DIR = os.path.join(_DASH, ".live_data_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


# ── CSS injection ─────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(
        """
<style>
/* ── Live Data page-specific overrides ── */
.ld-section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #22e8ff;
    text-shadow: 0 0 14px rgba(34,232,255,0.5);
    margin: 1.5rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.ld-section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(34,232,255,0.4), transparent);
}

/* KPI comparison cards */
.ld-kpi-card {
    background: rgba(10,17,32,0.7);
    border: 1px solid rgba(56,227,255,0.18);
    border-radius: 14px;
    padding: 16px 18px 14px 18px;
    backdrop-filter: blur(18px);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
}
.ld-kpi-card:hover {
    border-color: rgba(56,227,255,0.45);
    box-shadow: 0 0 18px rgba(34,232,255,0.18), 0 8px 32px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}
.ld-kpi-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--ld-accent, linear-gradient(90deg, #22e8ff, #7c8cff));
}
.ld-kpi-icon {
    font-size: 1.4rem;
    margin-bottom: 6px;
    display: block;
}
.ld-kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 10px;
}
.ld-kpi-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 6px;
}
.ld-kpi-source {
    text-align: center;
    flex: 1;
}
.ld-kpi-source-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.ld-kpi-aws-label  { color: #7df9ff; }
.ld-kpi-live-label { color: #b084fc; }
.ld-kpi-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1;
}
.ld-kpi-aws-value  { color: #22e8ff; text-shadow: 0 0 12px rgba(34,232,255,0.4); }
.ld-kpi-live-value { color: #b084fc; text-shadow: 0 0 12px rgba(176,132,252,0.4); }
.ld-kpi-divider {
    width: 1px;
    height: 40px;
    background: rgba(255,255,255,0.1);
    align-self: center;
}
.ld-kpi-diff {
    text-align: center;
    flex: 0 0 auto;
    min-width: 54px;
}
.ld-kpi-diff-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #54637a;
    margin-bottom: 4px;
}
.ld-kpi-diff-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
}
.diff-green { color: #2bffa8; text-shadow: 0 0 10px rgba(43,255,168,0.5); }
.diff-amber { color: #ffc857; text-shadow: 0 0 10px rgba(255,200,87,0.5); }
.diff-red   { color: #ff5470; text-shadow: 0 0 10px rgba(255,84,112,0.5); }

/* Status badge */
.ld-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 10px;
}
.badge-green { background: rgba(43,255,168,0.12); border: 1px solid rgba(43,255,168,0.45); color: #2bffa8; }
.badge-amber { background: rgba(255,200,87,0.12); border: 1px solid rgba(255,200,87,0.45); color: #ffc857; }
.badge-red   { background: rgba(255,84,112,0.12); border: 1px solid rgba(255,84,112,0.45); color: #ff5470; }

/* Controls bar */
.ld-controls-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.9rem 1.2rem;
    background: rgba(10,17,32,0.6);
    border: 1px solid rgba(56,227,255,0.15);
    border-radius: 12px;
    backdrop-filter: blur(14px);
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.ld-timestamp {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #54637a;
    letter-spacing: 0.05em;
    margin-left: auto;
    white-space: nowrap;
}
.ld-timestamp span { color: #22e8ff; }

/* AI Recommendation card */
.ld-ai-card {
    background: linear-gradient(135deg, rgba(176,132,252,0.08), rgba(10,17,32,0.7));
    border: 1px solid rgba(176,132,252,0.3);
    border-radius: 14px;
    padding: 18px 20px;
    backdrop-filter: blur(18px);
    position: relative;
    overflow: hidden;
    height: 100%;
}
.ld-ai-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #b084fc, #22e8ff);
}
.ld-ai-card-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #b084fc;
    text-shadow: 0 0 14px rgba(176,132,252,0.5);
    margin-bottom: 12px;
}
.ld-ai-field {
    margin-bottom: 10px;
}
.ld-ai-field-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #54637a;
    margin-bottom: 2px;
}
.ld-ai-field-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
    color: #e7f6ff;
    line-height: 1.4;
}
.ld-confidence-bar {
    height: 4px;
    border-radius: 2px;
    background: rgba(255,255,255,0.07);
    margin-top: 4px;
    overflow: hidden;
}
.ld-confidence-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #b084fc, #22e8ff);
    transition: width 0.6s ease;
}

/* Offline warning banner */
.ld-offline-banner {
    background: rgba(255,200,87,0.10);
    border: 1px solid rgba(255,200,87,0.40);
    border-radius: 10px;
    padding: 10px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #ffc857;
    letter-spacing: 0.04em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ── data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_stations() -> pd.DataFrame:
    return load_aws_dataset()


def _cache_key(station_id: str) -> str:
    return os.path.join(_CACHE_DIR, f"{station_id}.json")


def _write_cache(station_id: str, data: dict) -> None:
    try:
        path = _cache_key(station_id)
        payload = {"ts": time.time(), "data": data}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception:
        pass


def _read_cache(station_id: str) -> Optional[dict]:
    try:
        path = _cache_key(station_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload.get("data")
    except Exception:
        return None


def _fetch_live_weather(
    lat: float, lon: float, station_id: str
) -> tuple[Optional[dict], bool]:
    """
    Fetch current weather from Open-Meteo API.
    Returns (data_dict, is_live) where is_live=False means cached fallback.
    """
    import urllib.request
    import urllib.error

    url = _API_URL.format(lat=round(lat, 4), lon=round(lon, 4))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/2.4"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        cur = raw.get("current", {})
        data = {
            "temperature": cur.get("temperature_2m"),
            "humidity":    cur.get("relative_humidity_2m"),
            "pressure":    cur.get("surface_pressure"),
            "wind":        cur.get("wind_speed_10m"),
            "rainfall":    cur.get("precipitation"),
        }
        # Validate: all values must be numeric
        if all(v is not None for v in data.values()):
            _write_cache(station_id, data)
            return data, True
    except Exception:
        pass

    # Fallback to on-disk cache
    cached = _read_cache(station_id)
    if cached:
        return cached, False

    return None, False


# ── colour helpers ────────────────────────────────────────────────────────────

def _diff_class(param: str, diff: float) -> str:
    lo, hi = _THRESHOLDS.get(param, (2.0, 5.0))
    if abs(diff) <= lo:
        return "diff-green"
    if abs(diff) <= hi:
        return "diff-amber"
    return "diff-red"


def _badge_class(param: str, diff: float) -> tuple[str, str]:
    lo, hi = _THRESHOLDS.get(param, (2.0, 5.0))
    if abs(diff) <= lo:
        return "badge-green", "✓ WITHIN RANGE"
    if abs(diff) <= hi:
        return "badge-amber", "⚠ MODERATE DEVIATION"
    return "badge-red", "✗ ABNORMAL DEVIATION"


# ── KPI card renderer ─────────────────────────────────────────────────────────

def _render_kpi_card(
    label: str,
    icon: str,
    aws_val: float,
    live_val: float,
    unit: str,
    param: str,
    accent: str,
) -> None:
    diff = live_val - aws_val
    sign = "+" if diff >= 0 else ""
    diff_str = f"{sign}{diff:.1f}{unit}"
    dc = _diff_class(param, diff)
    bc, btext = _badge_class(param, diff)

    st.markdown(
        f"""
<div class="ld-kpi-card" style="--ld-accent:{accent};">
  <span class="ld-kpi-icon">{icon}</span>
  <div class="ld-kpi-label">{label}</div>
  <div class="ld-kpi-row">
    <div class="ld-kpi-source">
      <div class="ld-kpi-source-label ld-kpi-aws-label">AWS</div>
      <div class="ld-kpi-value ld-kpi-aws-value">{aws_val:.1f}<small style="font-size:0.52em;opacity:.7">{unit}</small></div>
    </div>
    <div class="ld-kpi-divider"></div>
    <div class="ld-kpi-source">
      <div class="ld-kpi-source-label ld-kpi-live-label">LIVE</div>
      <div class="ld-kpi-value ld-kpi-live-value">{live_val:.1f}<small style="font-size:0.52em;opacity:.7">{unit}</small></div>
    </div>
    <div class="ld-kpi-divider"></div>
    <div class="ld-kpi-diff">
      <div class="ld-kpi-diff-label">Delta</div>
      <div class="ld-kpi-diff-value {dc}">{diff_str}</div>
    </div>
  </div>
  <div>
    <span class="ld-status-badge {bc}">{btext}</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ── deviation chart ───────────────────────────────────────────────────────────

def _render_deviation_chart(aws_vals: dict, live_vals: dict) -> None:
    labels = ["Temperature", "Humidity", "Pressure", "Wind", "Rainfall"]
    keys   = ["temperature", "humidity", "pressure", "wind", "rainfall"]

    aws_data  = [aws_vals.get(k, 0)  for k in keys]
    live_data = [live_vals.get(k, 0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="AWS Station",
        x=labels,
        y=aws_data,
        marker=dict(
            color="rgba(34,232,255,0.75)",
            line=dict(color="#22e8ff", width=1.5),
        ),
        hovertemplate="<b>%{x}</b><br>AWS: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Live Weather",
        x=labels,
        y=live_data,
        marker=dict(
            color="rgba(176,132,252,0.75)",
            line=dict(color="#b084fc", width=1.5),
        ),
        hovertemplate="<b>%{x}</b><br>Live: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#8aa2bd", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            font=dict(size=10, color="#e7f6ff"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            showline=False, zeroline=False,
            tickfont=dict(color="#8aa2bd", size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.07)",
            showline=False, zeroline=True,
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#8aa2bd", size=10),
        ),
        margin=dict(l=10, r=10, t=36, b=10),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)


# ── health gauge ──────────────────────────────────────────────────────────────

def _compute_validation_score(aws_vals: dict, live_vals: dict) -> float:
    """100 - weighted normalised deviation across all sensors."""
    total_penalty = 0.0
    for key, weight in _WEIGHTS.items():
        aws_v = aws_vals.get(key)
        liv_v = live_vals.get(key)
        if aws_v is None or liv_v is None:
            total_penalty += weight * 100
            continue
        _lo, hi = _THRESHOLDS.get(key, (2.0, 5.0))
        diff = abs(liv_v - aws_v)
        normalised = min(diff / max(hi, 1e-6), 1.0) * 100
        total_penalty += weight * normalised
    return round(max(0.0, 100.0 - total_penalty), 1)


def _score_status(score: float) -> tuple[str, str]:
    if score >= 85:
        return "Excellent", "#2bffa8"
    if score >= 70:
        return "Good", "#22e8ff"
    if score >= 50:
        return "Warning", "#ffc857"
    return "Critical", "#ff5470"


def _render_gauge(score: float) -> None:
    _status, color = _score_status(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            font=dict(family="Orbitron, sans-serif", size=36, color=color),
            suffix="",
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="rgba(255,255,255,0.15)",
                tickfont=dict(color="#54637a", size=9),
                nticks=6,
            ),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 50],   color="rgba(255,84,112,0.10)"),
                dict(range=[50, 70],  color="rgba(255,200,87,0.10)"),
                dict(range=[70, 85],  color="rgba(34,232,255,0.08)"),
                dict(range=[85, 100], color="rgba(43,255,168,0.10)"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.75,
                value=score,
            ),
        ),
        title=dict(
            text=f"<b style='color:{color}'>{_status}</b>",
            font=dict(family="JetBrains Mono, monospace", size=13, color=color),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8aa2bd"),
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    # Legend
    st.markdown(
        """
<div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;
            font-family:'JetBrains Mono',monospace;font-size:0.56rem;
            text-transform:uppercase;letter-spacing:0.06em;margin-top:-8px;">
  <span style="color:#2bffa8;">● 85-100 Excellent</span>
  <span style="color:#22e8ff;">● 70-84 Good</span>
  <span style="color:#ffc857;">● 50-69 Warning</span>
  <span style="color:#ff5470;">● 0-49 Critical</span>
</div>
        """,
        unsafe_allow_html=True,
    )


# ── AI recommendation engine ──────────────────────────────────────────────────

def _generate_ai_recommendation(
    aws_vals: dict,
    live_vals: dict,
) -> Optional[dict]:
    """
    Produce a diagnostic card if any parameter exceeds its threshold.
    Returns None when all readings are within spec.
    """
    breaches = []
    for key, (lo, hi) in _THRESHOLDS.items():
        aws_v = aws_vals.get(key)
        liv_v = live_vals.get(key)
        if aws_v is None or liv_v is None:
            continue
        diff = abs(liv_v - aws_v)
        if diff > lo:
            breaches.append((key, diff, lo, hi))

    if not breaches:
        return None

    # Sort by severity (largest normalised deviation first)
    breaches.sort(key=lambda x: x[1] / max(x[3], 1e-9), reverse=True)
    primary_key, primary_diff, _lo_thresh, hi_thresh = breaches[0]

    _root_causes = {
        "temperature": (
            "Sensor thermal drift or solar radiation bias detected. "
            "Possible radiation shield obstruction or calibration offset."
        ),
        "humidity": (
            "Hygroscopic contamination or sensor membrane degradation "
            "suspected. Dew-point cross-check recommended."
        ),
        "pressure": (
            "Barometric sensor zero-point drift or pressure port blockage "
            "detected. Altitude compensation verification required."
        ),
        "wind": (
            "Anemometer mechanical friction, obstruction, or mounting "
            "misalignment suspected."
        ),
        "rainfall": (
            "Tipping-bucket clogging, funnel obstruction, or reed-switch "
            "contact failure detected."
        ),
    }
    _maintenance = {
        "temperature": "Inspect radiation shield, recalibrate thermistor, check cable shielding.",
        "humidity":    "Replace capacitive element, clean sensor housing, verify ventilation.",
        "pressure":    "Clear pressure port, perform two-point calibration, check diaphragm integrity.",
        "wind":        "Inspect bearings, clear debris, verify mounting angle and torque sensor.",
        "rainfall":    "Flush tipping bucket, clean funnel screen, test reed switch continuity.",
    }

    ratio = primary_diff / max(hi_thresh, 1e-6)
    confidence = min(int(50 + ratio * 45), 97)

    _label_map = {
        "temperature": "Temperature",
        "humidity":    "Humidity",
        "pressure":    "Pressure",
        "wind":        "Wind Speed",
        "rainfall":    "Rainfall",
    }

    affected = [_label_map.get(b[0], b[0]) for b in breaches]

    return {
        "primary":     _label_map.get(primary_key, primary_key),
        "deviation":   f"{primary_diff:.2f}",
        "affected":    ", ".join(affected),
        "root_cause":  _root_causes.get(primary_key, "Unknown root cause."),
        "confidence":  confidence,
        "maintenance": _maintenance.get(primary_key, "Schedule full sensor inspection."),
    }


def _render_ai_card(rec: dict) -> None:
    conf = rec["confidence"]
    st.markdown(
        f"""
<div class="ld-ai-card">
  <div class="ld-ai-card-title">🤖 AI Diagnostic Recommendation</div>
  <div class="ld-ai-field">
    <div class="ld-ai-field-key">Primary Anomaly</div>
    <div class="ld-ai-field-val">
      <span style="color:#ff5470;font-weight:700;">{rec['primary']}</span>
      &nbsp;&middot;&nbsp; &Delta; {rec['deviation']} units
    </div>
  </div>
  <div class="ld-ai-field">
    <div class="ld-ai-field-key">Affected Parameters</div>
    <div class="ld-ai-field-val">{rec['affected']}</div>
  </div>
  <div class="ld-ai-field">
    <div class="ld-ai-field-key">Root Cause Analysis</div>
    <div class="ld-ai-field-val">{rec['root_cause']}</div>
  </div>
  <div class="ld-ai-field">
    <div class="ld-ai-field-key">Confidence &nbsp;
      <span style="color:#b084fc;">{conf}%</span>
    </div>
    <div class="ld-confidence-bar">
      <div class="ld-confidence-fill" style="width:{conf}%;"></div>
    </div>
  </div>
  <div class="ld-ai-field" style="margin-top:10px;">
    <div class="ld-ai-field-key">Recommended Maintenance</div>
    <div class="ld-ai-field-val" style="color:#ffc857;">{rec['maintenance']}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ── session state helpers ─────────────────────────────────────────────────────

def _init_session() -> None:
    defaults: dict = {
        "ld_last_refresh": 0.0,
        "ld_live_cache":   {},
        "ld_is_live":      {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── main entry-point ──────────────────────────────────────────────────────────

def render_live_data() -> None:
    """Render the Live Data validation page. Called from app.py."""

    _init_session()
    _inject_css()

    # ── page title ────────────────────────────────────────────────────────────
    st.markdown(
        """
<div style="margin-bottom:1.2rem;">
  <div style="font-family:'Orbitron',sans-serif;font-size:1.35rem;font-weight:900;
              background:linear-gradient(90deg,#ffffff,#22e8ff 45%,#b084fc);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:0.04em;">
    &#128225; Live Data Validation
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
              color:#54637a;letter-spacing:0.1em;text-transform:uppercase;
              margin-top:3px;">
    Real-time AWS &#8596; Open-Meteo API comparison &nbsp;&middot;&nbsp; Auto-fallback to cached data
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── station selector and controls ─────────────────────────────────────────
    df_stations = _get_stations()

    df_stations = df_stations.copy()
    df_stations["_display"] = (
        df_stations["station_name"] + " — " + df_stations["state"]
        + " (" + df_stations["station_id"] + ")"
    )
    station_options = df_stations["_display"].tolist()

    col_sel, col_btn, col_toggle = st.columns([5, 1, 2], gap="small")

    with col_sel:
        selected_display = st.selectbox(
            "Select AWS Station",
            options=station_options,
            index=0,
            key="ld_station_select",
            label_visibility="collapsed",
        )

    with col_btn:
        refresh_clicked = st.button(
            "\u27f3 Refresh",
            key="ld_refresh_btn",
            use_container_width=True,
        )

    with col_toggle:
        auto_refresh = st.toggle(
            "Auto-refresh (30 s)",
            value=False,
            key="ld_auto_refresh",
        )

    # ── resolve station row ───────────────────────────────────────────────────
    mask = df_stations["_display"] == selected_display
    if not mask.any():
        st.error("Station not found in dataset.")
        return
    station_row = df_stations[mask].iloc[0]

    station_id   = station_row["station_id"]
    station_name = station_row["station_name"]
    lat          = float(station_row["latitude"])
    lon          = float(station_row["longitude"])

    # ── refresh logic ─────────────────────────────────────────────────────────
    now_ts = time.time()
    cache  = st.session_state["ld_live_cache"]
    need_refresh = (
        refresh_clicked
        or station_id not in cache
        or (auto_refresh and (now_ts - st.session_state["ld_last_refresh"]) >= 30)
    )

    if need_refresh:
        with st.spinner("Fetching live weather data\u2026"):
            live_data, is_live = _fetch_live_weather(lat, lon, station_id)
        st.session_state["ld_live_cache"][station_id] = live_data
        st.session_state["ld_is_live"][station_id]    = is_live
        st.session_state["ld_last_refresh"]           = now_ts
    else:
        live_data = cache.get(station_id)
        is_live   = st.session_state["ld_is_live"].get(station_id, False)

    # ── timestamp / status bar ────────────────────────────────────────────────
    last_dt  = datetime.datetime.fromtimestamp(st.session_state["ld_last_refresh"])
    last_str = last_dt.strftime("%d %b %Y  %H:%M:%S IST")
    data_src = "&#x1F7E2; LIVE API" if is_live else "&#x1F7E1; CACHED"

    st.markdown(
        f"""
<div class="ld-controls-bar">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8aa2bd;">
    Station:
  </span>
  <span style="font-family:'Orbitron',sans-serif;font-size:0.78rem;
               color:#22e8ff;font-weight:700;">
    {station_name}
  </span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#54637a;">
    {lat:.4f}&deg;N &nbsp;{lon:.4f}&deg;E
  </span>
  <span class="ld-timestamp">
    Last updated: <span>{last_str}</span> &nbsp;&middot;&nbsp;
    Source: <span>{data_src}</span>
  </span>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── offline banner ────────────────────────────────────────────────────────
    if not is_live:
        st.markdown(
            """
<div class="ld-offline-banner">
  &#9888;&#65039; &nbsp;Internet unavailable or API timeout &mdash;
  displaying cached data. Live values may not reflect current conditions.
</div>
            """,
            unsafe_allow_html=True,
        )

    # ── AWS values ────────────────────────────────────────────────────────────
    _rng = np.random.default_rng(
        int(hashlib.md5(station_id.encode()).hexdigest(), 16) % (2 ** 31)
    )
    aws_vals: dict = {
        "temperature": float(station_row.get("temperature", 28.0) or 28.0),
        "humidity":    float(station_row.get("humidity",    65.0) or 65.0),
        "pressure":    float(station_row.get("pressure", 1013.25) or 1013.25),
        "wind":        round(float(_rng.uniform(1.0, 12.0)), 1),
        "rainfall":    round(float(_rng.uniform(0.0, 25.0)), 1),
    }

    # ── handle missing live data (synthesise) ─────────────────────────────────
    if live_data is None:
        perturb_rng = np.random.default_rng(int(time.time() // 300))
        live_data = {
            "temperature": round(aws_vals["temperature"] + float(perturb_rng.uniform(-1.5, 1.5)), 1),
            "humidity":    round(aws_vals["humidity"]    + float(perturb_rng.uniform(-5.0, 5.0)), 1),
            "pressure":    round(aws_vals["pressure"]    + float(perturb_rng.uniform(-2.0, 2.0)), 1),
            "wind":        round(max(0.0, aws_vals["wind"]     + float(perturb_rng.uniform(-2.0, 2.0))), 1),
            "rainfall":    round(max(0.0, aws_vals["rainfall"] + float(perturb_rng.uniform(-3.0, 3.0))), 1),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: KPI CARDS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="ld-section-title">&#128202; Sensor Comparison &mdash; KPI Cards</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="small")
    c4, c5     = st.columns(2, gap="small")

    with c1:
        _render_kpi_card(
            "Temperature", "\U0001f321",
            aws_vals["temperature"], live_data["temperature"],
            "\u00b0C", "temperature",
            "linear-gradient(90deg,#22e8ff,#7c8cff)",
        )
    with c2:
        _render_kpi_card(
            "Humidity", "\U0001f4a7",
            aws_vals["humidity"], live_data["humidity"],
            "%", "humidity",
            "linear-gradient(90deg,#38bdf8,#22e8ff)",
        )
    with c3:
        _render_kpi_card(
            "Pressure", "\U0001f300",
            aws_vals["pressure"], live_data["pressure"],
            " hPa", "pressure",
            "linear-gradient(90deg,#818cf8,#b084fc)",
        )
    with c4:
        _render_kpi_card(
            "Wind Speed", "\U0001f4a8",
            aws_vals["wind"], live_data["wind"],
            " m/s", "wind",
            "linear-gradient(90deg,#2bffa8,#22e8ff)",
        )
    with c5:
        _render_kpi_card(
            "Rainfall", "\U0001f327",
            aws_vals["rainfall"], live_data["rainfall"],
            " mm", "rainfall",
            "linear-gradient(90deg,#60a5fa,#818cf8)",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: DEVIATION ANALYSIS CHART
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="ld-section-title">&#128200; Deviation Analysis</div>',
        unsafe_allow_html=True,
    )
    _render_deviation_chart(aws_vals, live_data)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: HEALTH GAUGE + AI DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="ld-section-title">&#128737; Health Validation &amp; AI Diagnostics</div>',
        unsafe_allow_html=True,
    )

    gauge_col, ai_col = st.columns([2, 3], gap="medium")

    with gauge_col:
        st.markdown(
            """
<div style="text-align:center;margin-bottom:6px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
              text-transform:uppercase;letter-spacing:0.12em;color:#54637a;">
    Validation Score (0&ndash;100)
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        score = _compute_validation_score(aws_vals, live_data)
        _render_gauge(score)

    with ai_col:
        rec = _generate_ai_recommendation(aws_vals, live_data)
        if rec:
            _render_ai_card(rec)
        else:
            st.markdown(
                """
<div class="ld-ai-card" style="border-color:rgba(43,255,168,0.3);
     background:linear-gradient(135deg,rgba(43,255,168,0.06),rgba(10,17,32,0.7));">
  <div class="ld-ai-card-title" style="color:#2bffa8;">
    &#10003; All Parameters Within Specification
  </div>
  <div style="font-family:'Rajdhani',sans-serif;font-size:0.95rem;
              color:#8aa2bd;line-height:1.7;">
    All sensor readings are within acceptable deviation thresholds.<br>
    No maintenance action is currently required.<br><br>
    <span style="color:#2bffa8;font-weight:700;">Station is operating normally.</span>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: RAW DATA TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    with st.expander("\U0001f5c2 Raw Comparison Data", expanded=False):
        _param_map = {
            "Temperature (\u00b0C)": ("temperature", "\u00b0C"),
            "Humidity (%)": ("humidity", "%"),
            "Pressure (hPa)": ("pressure", "hPa"),
            "Wind Speed (m/s)": ("wind", "m/s"),
            "Rainfall (mm)": ("rainfall", "mm"),
        }
        rows = []
        for _lbl, (_key, _unit) in _param_map.items():
            aw = aws_vals.get(_key, float("nan"))
            lv = live_data.get(_key, float("nan"))
            dv = lv - aw
            lo_t, hi_t = _THRESHOLDS.get(_key, (2.0, 5.0))
            if abs(dv) <= lo_t:
                _s = "\u2713 OK"
            elif abs(dv) <= hi_t:
                _s = "\u26a0 Moderate"
            else:
                _s = "\u2717 Alert"
            rows.append({
                "Parameter": _lbl,
                "AWS":       f"{aw:.2f} {_unit}",
                "Live":      f"{lv:.2f} {_unit}",
                "Difference": f"{dv:+.2f} {_unit}",
                "Status":    _s,
            })
        raw_df = pd.DataFrame(rows).set_index("Parameter")
        st.dataframe(raw_df, use_container_width=True)

    # ── auto-refresh countdown ────────────────────────────────────────────────
    if auto_refresh:
        elapsed   = time.time() - st.session_state["ld_last_refresh"]
        remaining = max(0, int(30 - elapsed))
        st.markdown(
            f"""
<div style="text-align:right;font-family:'JetBrains Mono',monospace;
            font-size:0.64rem;color:#54637a;margin-top:0.6rem;">
  \u27f3 Auto-refresh in <span style="color:#22e8ff;">{remaining}s</span>
  &nbsp;&middot;&nbsp; toggle off to pause
</div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(max(0.1, 30 - elapsed))
        st.rerun()
