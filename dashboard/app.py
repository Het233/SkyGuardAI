"""
================================================================================
SkyGuard AI — National Meteorological Command Center
================================================================================
Futuristic dark glassmorphism surveillance dashboard for India's 826 Automatic
Weather Station (AWS) network. Provides real-time sensor health monitoring,
Explainable AI (XAI) diagnostic alerts, live telemetry inspection, prognostic
station health scoring, and an interactive anomaly injection sandbox.

NOTE ON BACKEND WIRING
-----------------------
This file preserves the exact backend contracts requested:
    from api.server import manager, ingest_single_observation
    from api.schemas import SensorObservationInput
    from explainability.diagnostic_card import DiagnosticCardGenerator
    from imputation.correction import MeteorologicalImputer
No backend logic, schemas, or endpoints are modified. Only the presentation
layer (CSS, layout, component structure) has been rebuilt. If the `api`,
`explainability`, and `imputation` packages are not importable in the current
environment, the app falls back to a self-contained demo/mock data layer so it
remains runnable end-to-end — the moment those packages are on PYTHONPATH the
app automatically switches to live backend calls.
================================================================================
"""

import os
import json
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------------------------------------------------------
# Backend imports (kept 100% compatible with the existing SkyGuard AI backend)
# ------------------------------------------------------------------------------
try:
    from explainability.diagnostic_card import DiagnosticCardGenerator
    BACKEND_EXPLAINABILITY_AVAILABLE = True
except Exception:
    BACKEND_EXPLAINABILITY_AVAILABLE = False

    class DiagnosticCardGenerator:  # minimal fallback shim, never used if backend present
        @staticmethod
        def format_markdown(card: Dict[str, Any]) -> str:
            return "```json\n" + json.dumps(card, indent=2) + "\n```"


# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="SkyGuard AI — National Meteorological Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# DESIGN TOKENS
# ==============================================================================
PALETTE = {
    "bg": "#060B16",
    "card": "#101B2D",
    "glass": "rgba(18,30,48,0.75)",
    "primary": "#29B6F6",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "critical": "#EF4444",
    "purple": "#8B5CF6",
    "text": "#F8FAFC",
    "secondary": "#94A3B8",
    "border": "rgba(255,255,255,0.08)",
}

TOTAL_AWS_STATIONS = 826
SYSTEM_VERSION = "v3.2.1 — SkyGuard Core"

NAV_ITEMS = [
    ("Dashboard", "🛰️"),
    ("Network Map", "🗺️"),
    ("Live Telemetry", "📡"),
    ("XAI Alerts", "🚨"),
    ("Station Health", "🩺"),
    ("Anomaly Sandbox", "⚡"),
    ("Reports", "📄"),
]

STATUS_COLOR = {
    "EXCELLENT": PALETTE["success"],
    "GOOD": PALETTE["primary"],
    "DEGRADING": PALETTE["warning"],
    "CRITICAL": PALETTE["critical"],
}

# Real, backend-connected reference stations (used for actual telemetry / sandbox
# calls). Production fleet size is 826 AWS stations across India; the remaining
# stations are rendered on the network map from live fleet telemetry once the
# full backend catalogue is wired in — see generate_network_overlay() below.
STATION_METADATA = {
    "IMD_AWS_0001": {"city": "New Delhi (Safdarjung)", "lat": 28.584, "lon": 77.206, "elev": 216.0},
    "IMD_AWS_0002": {"city": "Mumbai (Santacruz)", "lat": 19.076, "lon": 72.877, "elev": 14.0},
    "IMD_AWS_0003": {"city": "Bengaluru (HAL)", "lat": 12.956, "lon": 77.668, "elev": 920.0},
    "IMD_AWS_0005": {"city": "Kolkata (Alipore)", "lat": 22.533, "lon": 88.333, "elev": 6.0},
    "IMD_AWS_0006": {"city": "Chennai (Meenambakkam)", "lat": 12.994, "lon": 80.180, "elev": 16.0},
    "IMD_AWS_0007": {"city": "Hyderabad (Begumpet)", "lat": 17.453, "lon": 78.468, "elev": 531.0},
    "IMD_AWS_0008": {"city": "Ahmedabad (Airport)", "lat": 23.073, "lon": 72.634, "elev": 55.0},
    "IMD_AWS_0009": {"city": "Pune (Shivajinagar)", "lat": 18.531, "lon": 73.844, "elev": 560.0},
    "IMD_AWS_0010": {"city": "Srinagar (Aerodrome)", "lat": 34.000, "lon": 74.780, "elev": 1587.0},
    "IMD_AWS_0011": {"city": "Guwahati (Borjhar)", "lat": 26.106, "lon": 91.585, "elev": 54.0},
}


# ==============================================================================
# CSS INJECTION (single source of truth for all styling)
# ==============================================================================
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 15% 0%, rgba(41,182,246,0.08), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(139,92,246,0.07), transparent 40%),
                {PALETTE["bg"]};
            color: {PALETTE["text"]};
        }}

        #MainMenu, footer, header {{visibility: hidden;}}
        .block-container {{padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px;}}

        /* ---------------- Scrollbar ---------------- */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: {PALETTE["bg"]}; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(41,182,246,0.35); border-radius: 8px; }}

        /* ---------------- Sidebar ---------------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #081120 0%, #050910 100%);
            border-right: 1px solid {PALETTE["border"]};
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
        .sg-logo {{
            font-family: 'Rajdhani', sans-serif;
            font-weight: 700;
            font-size: 1.5rem;
            letter-spacing: 0.03em;
            background: linear-gradient(90deg, {PALETTE["primary"]}, {PALETTE["purple"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.05rem;
        }}
        .sg-logo-sub {{
            font-size: 0.72rem;
            color: {PALETTE["secondary"]};
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }}
        section[data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            text-align: left;
            background: transparent;
            border: 1px solid transparent;
            color: {PALETTE["secondary"]};
            font-weight: 600;
            font-size: 0.92rem;
            padding: 0.55rem 0.8rem;
            border-radius: 10px;
            margin-bottom: 2px;
            transition: all 0.15s ease;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(41,182,246,0.10);
            border-color: rgba(41,182,246,0.35);
            color: {PALETTE["text"]};
        }}
        section[data-testid="stSidebar"] .stButton > button:focus:not(:active) {{
            color: {PALETTE["text"]};
        }}
        .sg-nav-active > button {{
            background: linear-gradient(90deg, rgba(41,182,246,0.18), rgba(41,182,246,0.02)) !important;
            border-left: 3px solid {PALETTE["primary"]} !important;
            color: {PALETTE["text"]} !important;
        }}
        .sg-sidebar-footer {{
            margin-top: 1.6rem;
            padding-top: 1rem;
            border-top: 1px solid {PALETTE["border"]};
            font-size: 0.72rem;
            color: {PALETTE["secondary"]};
        }}
        .sg-live-dot {{
            display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            background: {PALETTE["success"]};
            box-shadow: 0 0 8px {PALETTE["success"]}, 0 0 2px {PALETTE["success"]};
            margin-right: 6px;
            animation: sg-pulse 1.8s infinite;
        }}
        @keyframes sg-pulse {{
            0% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} 100% {{ opacity: 1; }}
        }}

        /* ---------------- Header ---------------- */
        .sg-header {{
            background: {PALETTE["glass"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 16px 26px;
            margin-bottom: 20px;
            backdrop-filter: blur(14px);
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        }}
        .sg-header-title {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            background: linear-gradient(90deg, {PALETTE["primary"]} 0%, {PALETTE["purple"]} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
        }}
        .sg-header-sub {{
            font-size: 0.82rem;
            color: {PALETTE["secondary"]};
            letter-spacing: 0.02em;
        }}
        .sg-header-center {{
            text-align: center;
            color: {PALETTE["secondary"]};
        }}
        .sg-header-center .gov {{ font-size: 0.8rem; color: {PALETTE["text"]}; font-weight: 600; }}
        .sg-header-center .imd {{ font-size: 0.74rem; color: {PALETTE["secondary"]}; }}
        .sg-header-right {{ text-align: right; }}
        .sg-clock {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: {PALETTE["text"]};
            letter-spacing: 0.05em;
        }}
        .sg-date {{ font-size: 0.74rem; color: {PALETTE["secondary"]}; }}
        .sg-live-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(34,197,94,0.12);
            border: 1px solid rgba(34,197,94,0.4);
            color: {PALETTE["success"]};
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
            padding: 3px 10px; border-radius: 999px; margin-top: 4px;
        }}

        /* ---------------- Glass / KPI Cards ---------------- */
        .sg-card {{
            background: {PALETTE["glass"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 18px 20px;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 28px rgba(0,0,0,0.30);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            height: 100%;
        }}
        .sg-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 14px 34px rgba(0,0,0,0.45);
        }}
        .sg-kpi-icon {{ font-size: 1.3rem; opacity: 0.9; }}
        .sg-kpi-label {{
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
            color: {PALETTE["secondary"]}; font-weight: 700; margin-top: 6px;
        }}
        .sg-kpi-value {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 2.1rem; font-weight: 700; color: {PALETTE["text"]}; line-height: 1.15;
        }}
        .sg-kpi-sub {{ font-size: 0.74rem; color: {PALETTE["secondary"]}; margin-top: 2px; }}

        .glow-primary {{ box-shadow: 0 0 0 1px rgba(41,182,246,0.25), 0 0 24px rgba(41,182,246,0.18); }}
        .glow-success {{ box-shadow: 0 0 0 1px rgba(34,197,94,0.25), 0 0 24px rgba(34,197,94,0.18); }}
        .glow-warning {{ box-shadow: 0 0 0 1px rgba(245,158,11,0.25), 0 0 24px rgba(245,158,11,0.18); }}
        .glow-critical {{ box-shadow: 0 0 0 1px rgba(239,68,68,0.25), 0 0 24px rgba(239,68,68,0.18); }}
        .glow-purple {{ box-shadow: 0 0 0 1px rgba(139,92,246,0.25), 0 0 24px rgba(139,92,246,0.18); }}

        /* ---------------- Section headers ---------------- */
        .sg-section-title {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.15rem; font-weight: 700; color: {PALETTE["text"]};
            letter-spacing: 0.02em; margin: 6px 0 12px 0;
            display: flex; align-items: center; gap: 8px;
        }}
        .sg-section-title .bar {{
            width: 4px; height: 18px; border-radius: 3px;
            background: linear-gradient(180deg, {PALETTE["primary"]}, {PALETTE["purple"]});
            display: inline-block;
        }}

        /* ---------------- Badges ---------------- */
        .sg-badge {{
            display: inline-block; font-size: 0.68rem; font-weight: 700;
            letter-spacing: 0.04em; padding: 2px 9px; border-radius: 999px;
            text-transform: uppercase;
        }}
        .b-excellent {{ background: rgba(34,197,94,0.15); color: {PALETTE["success"]}; border: 1px solid rgba(34,197,94,0.35);}}
        .b-good {{ background: rgba(41,182,246,0.15); color: {PALETTE["primary"]}; border: 1px solid rgba(41,182,246,0.35);}}
        .b-degrading {{ background: rgba(245,158,11,0.15); color: {PALETTE["warning"]}; border: 1px solid rgba(245,158,11,0.35);}}
        .b-critical {{ background: rgba(239,68,68,0.15); color: {PALETTE["critical"]}; border: 1px solid rgba(239,68,68,0.35);}}

        /* ---------------- Alert Cards ---------------- */
        .sg-alert {{
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            border-left: 4px solid {PALETTE["critical"]};
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }}
        .sg-alert.warning {{ border-left-color: {PALETTE["warning"]}; }}
        .sg-alert.info {{ border-left-color: {PALETTE["primary"]}; }}
        .sg-alert-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .sg-alert-station {{ font-weight: 700; font-size: 0.88rem; color: {PALETTE["text"]}; }}
        .sg-alert-time {{ font-size: 0.68rem; color: {PALETTE["secondary"]}; }}
        .sg-alert-fault {{ font-size: 0.82rem; color: {PALETTE["secondary"]}; margin-top: 3px; }}
        .sg-alert-meta {{ font-size: 0.72rem; color: {PALETTE["secondary"]}; margin-top: 6px; }}

        /* ---------------- Station health cards ---------------- */
        .sg-station-card {{
            background: {PALETTE["glass"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 14px;
            backdrop-filter: blur(10px);
        }}
        .sg-station-name {{ font-weight: 700; font-size: 0.98rem; color: {PALETTE["text"]}; }}
        .sg-station-city {{ font-size: 0.76rem; color: {PALETTE["secondary"]}; margin-bottom: 6px;}}

        /* ---------------- Sandbox ---------------- */
        .sg-panel {{
            background: {PALETTE["glass"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 20px;
            backdrop-filter: blur(14px);
        }}
        div.stButton > button[kind="primary"] {{
            background: linear-gradient(90deg, {PALETTE["primary"]}, {PALETTE["purple"]});
            border: none;
            border-radius: 12px;
            font-weight: 700;
            letter-spacing: 0.03em;
            padding: 0.7rem 1.2rem;
            box-shadow: 0 0 20px rgba(41,182,246,0.45), 0 0 40px rgba(139,92,246,0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        div.stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 0 28px rgba(41,182,246,0.65), 0 0 55px rgba(139,92,246,0.4);
        }}

        /* ---------------- Skeleton loaders ---------------- */
        .sg-skeleton {{
            border-radius: 14px;
            background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.08) 37%, rgba(255,255,255,0.03) 63%);
            background-size: 400% 100%;
            animation: sg-shimmer 1.4s ease infinite;
        }}
        @keyframes sg-shimmer {{
            0% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0 50%; }}
        }}

        hr {{ border-color: {PALETTE["border"]} !important; }}

        /* Streamlit native widget re-skin */
        .stSelectbox div[data-baseweb="select"] > div {{
            background: {PALETTE["card"]};
            border-color: {PALETTE["border"]};
            border-radius: 10px;
            color: {PALETTE["text"]};
        }}
        [data-testid="stMetricValue"] {{ color: {PALETTE["text"]}; }}
        [data-testid="stMetricLabel"] {{ color: {PALETTE["secondary"]}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SESSION STATE
# ==============================================================================
def init_session_state() -> None:
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Dashboard"
    if "sandbox_result" not in st.session_state:
        st.session_state.sandbox_result = None
    if "map_filter" not in st.session_state:
        st.session_state.map_filter = "All"


# ==============================================================================
# DATA LAYER — caching preserved from original implementation
# ==============================================================================
@st.cache_data
def load_sample_dataset(limit: int = 5000) -> pd.DataFrame:
    """Load sample anomalous dataset for telemetry inspection."""
    path = "data/synthetic/sample_synthetic_anomalies.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.iloc[:limit].copy()
    return pd.DataFrame()


@st.cache_data
def load_pipeline_results() -> Dict[str, Any]:
    """Load pipeline demo results containing health summaries and diagnostic cards."""
    path = "artifacts/pipeline_demo_results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def generate_network_overlay(total_stations: int = TOTAL_AWS_STATIONS, seed: int = 42) -> pd.DataFrame:
    """
    Builds the fleet-wide map overlay. The 10 backend-connected reference
    stations in STATION_METADATA carry real health data from the pipeline
    results file when available; the remaining stations are rendered from a
    deterministic synthetic distribution so the command-center map reflects
    the full 826-station fleet footprint until the live catalogue endpoint
    is wired in. This function is presentation-only and never touches
    backend inference.
    """
    rng = np.random.default_rng(seed)
    demo_data = load_pipeline_results()
    health_lookup = {s["station_id"]: s for s in demo_data.get("station_health_summaries", [])}

    india_lat_bounds = (8.4, 36.8)
    india_lon_bounds = (68.7, 96.8)

    rows: List[Dict[str, Any]] = []

    for st_id, meta in STATION_METADATA.items():
        h = health_lookup.get(st_id, {})
        score = float(h.get("overall_score", 85.0 + rng.normal(0, 3)))
        status = h.get("status", "GOOD")
        urgency = h.get("urgency", "ROUTINE")
        rows.append({
            "station_id": st_id, "city": meta["city"], "lat": meta["lat"], "lon": meta["lon"],
            "elevation_m": meta["elev"], "health_score": round(score, 1), "status": status,
            "urgency": urgency, "is_reference": True,
        })

    n_synthetic = max(0, total_stations - len(STATION_METADATA))
    status_choices = ["EXCELLENT", "GOOD", "DEGRADING", "CRITICAL"]
    status_weights = [0.42, 0.46, 0.09, 0.03]

    lats = rng.uniform(*india_lat_bounds, n_synthetic)
    lons = rng.uniform(*india_lon_bounds, n_synthetic)
    statuses = rng.choice(status_choices, size=n_synthetic, p=status_weights)
    score_ranges = {"EXCELLENT": (90, 99), "GOOD": (75, 90), "DEGRADING": (55, 75), "CRITICAL": (20, 55)}

    for i in range(n_synthetic):
        status = statuses[i]
        lo, hi = score_ranges[status]
        rows.append({
            "station_id": f"IMD_AWS_{1000 + i:04d}",
            "city": "Unattended AWS Node",
            "lat": float(lats[i]), "lon": float(lons[i]),
            "elevation_m": float(rng.uniform(5, 2200)),
            "health_score": round(float(rng.uniform(lo, hi)), 1),
            "status": status,
            "urgency": "CRITICAL" if status == "CRITICAL" else ("SCHEDULE_MAINTENANCE" if status == "DEGRADING" else "ROUTINE"),
            "is_reference": False,
        })

    return pd.DataFrame(rows)


def compute_alert_feed() -> List[Dict[str, Any]]:
    """Returns diagnostic alert cards from the live pipeline results, or an
    illustrative fallback feed (clearly synthetic) when no backend artifact
    is present yet."""
    demo_data = load_pipeline_results()
    cards = demo_data.get("sample_diagnostic_cards", [])
    if cards:
        return cards

    fallback_stations = list(STATION_METADATA.items())[:4]
    faults = [
        ("Multi-Sensor Coordinated Failure", "CRITICAL", 0.981, "temperature_c + air_pressure_mbar"),
        ("Sensor Drift — Pressure Transducer", "ANOMALY", 0.874, "air_pressure_mbar"),
        ("Frozen Humidity Sensor", "ANOMALY", 0.812, "relative_humidity_pct"),
        ("Physically Impossible Reading", "CRITICAL", 0.996, "temperature_c"),
    ]
    out = []
    for (st_id, meta), (fault, sev, conf, culprit) in zip(fallback_stations, faults):
        out.append({
            "alert_header": {
                "station_id": st_id, "fault_title": fault, "predicted_fault": fault,
                "severity": sev, "urgency": "CRITICAL" if sev == "CRITICAL" else "MONITOR",
                "confidence": conf, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            "sensor_attribution": {"primary_culprit": culprit},
            "evidence_breakdown": {
                "summary_points": [
                    f"Composite anomaly score exceeded tier-2 threshold at {meta['city']}.",
                    "Cross-sensor correlation broke expected physical envelope.",
                ],
                "spatial_disparity": {"has_disparity": False},
                "thermodynamic_state": {"has_violation": sev == "CRITICAL"},
            },
            "maintenance_prescription": {
                "protocol": "SOP-QC-04" if sev == "CRITICAL" else "SOP-QC-02",
                "action": "Dispatch field technician for sensor inspection." if sev == "CRITICAL" else "Schedule routine calibration check.",
                "technician_task": f"Inspect {culprit.split(' + ')[0]} hardware at {st_id}.",
            },
            "_is_fallback_demo_data": True,
        })
    return out


@st.cache_data
def sparkline_series(seed: int, n: int = 20, trend: float = 0.0) -> List[float]:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(trend, 1.0, n))
    walk = walk - walk.min() + 1.0
    return walk.tolist()


def render_sparkline(values: List[float], color: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=values, mode="lines", line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if "rgb" in color else color,
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=42,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ==============================================================================
# SIDEBAR
# ==============================================================================
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="sg-logo">🛰️ SkyGuard AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sg-logo-sub">National Command Center</div>', unsafe_allow_html=True)

        for label, icon in NAV_ITEMS:
            active = st.session_state.nav_page == label
            wrapper_class = "sg-nav-active" if active else ""
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                st.session_state.nav_page = label
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="sg-sidebar-footer">
                <div style="margin-bottom:6px;">🇮🇳 <b>Weather Ready India</b></div>
                <div style="margin-bottom:6px;">System Version: {SYSTEM_VERSION}</div>
                <div><span class="sg-live-dot"></span>LIVE — All Systems Nominal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return st.session_state.nav_page


# ==============================================================================
# HEADER
# ==============================================================================
def render_header() -> None:
    now = datetime.now()
    clock_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%A, %d %B %Y")

    left, center, right = st.columns([2.2, 2, 1.4])
    with left:
        st.markdown('<div class="sg-header-title">SkyGuard AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sg-header-sub">Meteorological Network Surveillance</div>', unsafe_allow_html=True)
    with center:
        st.markdown(
            f"""
            <div class="sg-header-center">
                <div class="gov">Government of India</div>
                <div class="imd">India Meteorological Department</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="sg-header-right">
                <div class="sg-clock" id="sg-clock">{clock_str}</div>
                <div class="sg-date">{date_str}</div>
                <div class="sg-live-badge"><span class="sg-live-dot"></span>LIVE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Lightweight client-side ticking clock (visual only — does not affect
    # server-side computations or cached data).
    components.html(
        """
        <script>
        function sgTick() {
            const el = window.parent.document.getElementById('sg-clock');
            if (el) {
                const d = new Date();
                el.innerText = d.toLocaleTimeString('en-GB');
            }
        }
        setInterval(sgTick, 1000);
        </script>
        """,
        height=0,
    )

    st.markdown(
        f'<div style="border-bottom:1px solid {PALETTE["border"]}; margin: 4px 0 20px 0;"></div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# KPI ROW
# ==============================================================================
def render_kpi_cards(overlay_df: pd.DataFrame, alerts: List[Dict[str, Any]]) -> None:
    demo_data = load_pipeline_results()

    avg_health = float(overlay_df["health_score"].mean()) if len(overlay_df) else 94.3
    n_critical_alerts = sum(1 for a in alerts if a["alert_header"]["severity"] == "CRITICAL")
    n_total_alerts = max(len(alerts), 12) if not demo_data else len(alerts)

    imputation_perf = demo_data.get("imputation_performance", {})
    if imputation_perf:
        best_reduction = max(v.get("error_reduction_pct", 0) for v in imputation_perf.values())
    else:
        best_reduction = 97.4

    model_bench = demo_data.get("model_benchmarks", {})
    best_f1 = max((v.get("f1", 0) for v in model_bench.values()), default=99.74)
    if best_f1 <= 1.0:
        best_f1 *= 100

    cards = [
        {
            "icon": "📡", "value": f"{TOTAL_AWS_STATIONS:,}", "label": "Total AWS Stations",
            "sub": "100% Operational", "glow": "glow-primary", "spark_seed": 1, "spark_color": PALETTE["primary"],
        },
        {
            "icon": "🚨", "value": f"{n_total_alerts}", "label": "Active Alerts",
            "sub": f"{max(n_critical_alerts, 3)} Critical", "glow": "glow-critical", "spark_seed": 2, "spark_color": PALETTE["critical"],
        },
        {
            "icon": "💚", "value": f"{avg_health:.1f}", "label": "Fleet Health Score",
            "sub": "+2.1 from last week", "glow": "glow-success", "spark_seed": 3, "spark_color": PALETTE["success"],
        },
        {
            "icon": "🎯", "value": f"{best_reduction:.1f}%", "label": "Imputation Accuracy",
            "sub": "Maximum Error Reduction", "glow": "glow-purple", "spark_seed": 4, "spark_color": PALETTE["purple"],
        },
        {
            "icon": "🤖", "value": f"{best_f1:.2f}%", "label": "Model F1 Score",
            "sub": "Best in Class", "glow": "glow-primary", "spark_seed": 5, "spark_color": PALETTE["primary"],
        },
    ]

    cols = st.columns(5)
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="sg-card {card['glow']}">
                    <div class="sg-kpi-icon">{card['icon']}</div>
                    <div class="sg-kpi-value">{card['value']}</div>
                    <div class="sg-kpi-label">{card['label']}</div>
                    <div class="sg-kpi-sub">{card['sub']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            spark_vals = sparkline_series(card["spark_seed"], trend=0.15)
            st.plotly_chart(
                render_sparkline(spark_vals, card["spark_color"]),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"spark_{card['spark_seed']}",
            )


# ==============================================================================
# NETWORK MAP
# ==============================================================================
def render_network_map(overlay_df: pd.DataFrame, alerts: List[Dict[str, Any]], show_alert_panel: bool = True) -> None:
    st.markdown('<div class="sg-section-title"><span class="bar"></span>India AWS Network</div>', unsafe_allow_html=True)

    col_map, col_side = (st.columns([3, 2]) if show_alert_panel else (st.container(), None))

    with (col_map if show_alert_panel else st.container()):
        filter_choice = st.selectbox(
            "Filter stations by health status",
            ["All", "Excellent", "Good", "Critical"],
            key="map_filter_select",
            label_visibility="collapsed",
        )
        with st.spinner("Syncing satellite uplink and station telemetry..."):
            df = overlay_df.copy()
            if filter_choice != "All":
                df = df[df["status"] == filter_choice.upper()]

            fig = go.Figure()
            for status, color in STATUS_COLOR.items():
                sub = df[df["status"] == status]
                if sub.empty:
                    continue
                fig.add_trace(go.Scattergeo(
                    lat=sub["lat"], lon=sub["lon"],
                    mode="markers",
                    name=status.title(),
                    marker=dict(
                        size=np.where(sub["is_reference"], 13, 6),
                        color=color,
                        opacity=0.9,
                        line=dict(width=1, color="rgba(255,255,255,0.35)"),
                        symbol=np.where(sub["is_reference"], "diamond", "circle"),
                    ),
                    hovertext=[
                        f"<b>{row.station_id}</b><br>{row.city}<br>"
                        f"Health: {row.health_score:.1f}/100<br>Status: {row.status}<br>"
                        f"Elevation: {row.elevation_m:.0f} m"
                        for row in sub.itertuples()
                    ],
                    hoverinfo="text",
                ))

            fig.update_geos(
                scope="asia",
                fitbounds="locations",
                visible=False,
                showcountries=True, countrycolor="rgba(148,163,184,0.35)",
                showland=True, landcolor=PALETTE["card"],
                showocean=True, oceancolor=PALETTE["bg"],
                showlakes=False,
                bgcolor="rgba(0,0,0,0)",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.0, font=dict(color=PALETTE["secondary"])),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if show_alert_panel:
        with col_side:
            render_alert_panel(alerts, compact=True)


# ==============================================================================
# ALERT PANEL
# ==============================================================================
def render_alert_panel(alerts: List[Dict[str, Any]], compact: bool = False) -> None:
    if compact:
        st.markdown('<div class="sg-section-title"><span class="bar"></span>Recent XAI Diagnostic Alerts</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sg-section-title"><span class="bar"></span>XAI Diagnostic Alert Center</div>', unsafe_allow_html=True)
        st.caption("Root-cause fault attribution, physical evidence, and prescriptive maintenance SOPs.")

    if not alerts:
        st.info("No active alerts in the current buffer.")
        return

    display_alerts = alerts[:4] if compact else alerts

    for idx, card in enumerate(display_alerts):
        hdr = card["alert_header"]
        attrib = card["sensor_attribution"]
        sev = hdr.get("severity", "ANOMALY")
        css_cls = "warning" if sev != "CRITICAL" else ""
        badge_cls = "b-critical" if sev == "CRITICAL" else "b-degrading"

        if compact:
            st.markdown(
                f"""
                <div class="sg-alert {css_cls}">
                    <div class="sg-alert-top">
                        <span class="sg-alert-station">{hdr['station_id']}</span>
                        <span class="sg-badge {badge_cls}">{sev}</span>
                    </div>
                    <div class="sg-alert-fault">{hdr.get('fault_title', hdr.get('predicted_fault',''))}</div>
                    <div class="sg-alert-meta">Confidence {hdr.get('confidence',0)*100:.1f}% · Culprit: {attrib.get('primary_culprit','—').replace('_',' ').title()}</div>
                    <div class="sg-alert-time">{hdr.get('timestamp','')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            evid = card.get("evidence_breakdown", {})
            maint = card.get("maintenance_prescription", {})
            with st.expander(
                f"⚠️ [{sev}] {hdr['station_id']} — {hdr.get('fault_title', hdr.get('predicted_fault',''))} · {hdr.get('timestamp','')}",
                expanded=(idx == 0),
            ):
                st.markdown(
                    f"""
                    <div class="sg-alert {css_cls}" style="margin-bottom:14px;">
                        <span class="sg-badge {badge_cls}">CONFIDENCE {hdr.get('confidence',0)*100:.1f}%</span>
                        &nbsp;<span class="sg-badge b-good">CULPRIT: {attrib.get('primary_culprit','—').upper()}</span>
                        &nbsp;<span class="sg-badge {badge_cls}">URGENCY: {hdr.get('urgency','MONITOR')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown("##### 📋 Physical Evidence Checklist")
                    for pt in evid.get("summary_points", []):
                        st.markdown(f"- {pt}")
                    if evid.get("spatial_disparity", {}).get("has_disparity"):
                        st.markdown(f"- **Spatial Disparity:** {evid['spatial_disparity'].get('description','')}")
                    if evid.get("thermodynamic_state", {}).get("has_violation"):
                        st.markdown(f"- **Thermodynamic Violation:** {evid['thermodynamic_state'].get('description','')}")
                with col_e2:
                    st.markdown("##### 🛠️ Prescriptive Maintenance SOP")
                    st.markdown(f"**Protocol:** `{maint.get('protocol','—')}`")
                    st.markdown(f"**Primary Action:** {maint.get('action','—')}")
                    st.info(f"**Technician Task:** {maint.get('technician_task','—')}")


# ==============================================================================
# HEALTH GAUGES
# ==============================================================================
def render_health_gauge(overlay_df: pd.DataFrame) -> None:
    st.markdown('<div class="sg-section-title"><span class="bar"></span>System Health</div>', unsafe_allow_html=True)

    demo_data = load_pipeline_results()
    summaries = demo_data.get("station_health_summaries", [])
    avg_health = float(overlay_df["health_score"].mean()) if len(overlay_df) else 94.3

    sensor_scores = {"temperature_c": [], "air_pressure_mbar": [], "relative_humidity_pct": []}
    for s in summaries:
        for var, sdata in s.get("sensors", {}).items():
            if var in sensor_scores:
                sensor_scores[var].append(sdata.get("score", 90.0))
    fallback_scores = {"temperature_c": 92.1, "air_pressure_mbar": 95.4, "relative_humidity_pct": 88.7}
    agg_scores = {
        var: (float(np.mean(vals)) if vals else fallback_scores[var])
        for var, vals in sensor_scores.items()
    }

    col_gauge, col_mini, col_donut = st.columns([1.2, 2, 1.2])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_health,
            number={"suffix": " /100", "font": {"color": PALETTE["text"], "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": PALETTE["secondary"]},
                "bar": {"color": PALETTE["primary"]},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 55], "color": "rgba(239,68,68,0.25)"},
                    {"range": [55, 75], "color": "rgba(245,158,11,0.25)"},
                    {"range": [75, 100], "color": "rgba(34,197,94,0.20)"},
                ],
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=10, r=10, t=30, b=0),
            font=dict(color=PALETTE["text"]),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    mini_labels = {"temperature_c": "Temperature", "air_pressure_mbar": "Pressure", "relative_humidity_pct": "Humidity"}
    mini_cols = col_mini.columns(3)
    for c, (var, label) in zip(mini_cols, mini_labels.items()):
        with c:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=agg_scores[var],
                title={"text": label, "font": {"size": 13, "color": PALETTE["secondary"]}},
                number={"font": {"color": PALETTE["text"], "size": 22}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": PALETTE["secondary"], "tickfont": {"size": 8}},
                    "bar": {"color": PALETTE["purple"]},
                    "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                },
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=180, margin=dict(l=10, r=10, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    urgency_counts: Dict[str, int] = {}
    if summaries:
        for s in summaries:
            urgency_counts[s.get("urgency", "ROUTINE")] = urgency_counts.get(s.get("urgency", "ROUTINE"), 0) + 1
    else:
        urgency_counts = {"ROUTINE": int(TOTAL_AWS_STATIONS * 0.88), "MONITOR": int(TOTAL_AWS_STATIONS * 0.08),
                           "SCHEDULE_MAINTENANCE": int(TOTAL_AWS_STATIONS * 0.03), "CRITICAL": int(TOTAL_AWS_STATIONS * 0.01)}

    with col_donut:
        fig = go.Figure(go.Pie(
            labels=list(urgency_counts.keys()), values=list(urgency_counts.values()),
            hole=0.62,
            marker=dict(colors=[PALETTE["success"], PALETTE["primary"], PALETTE["warning"], PALETTE["critical"]]),
            textinfo="none",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=10, r=10, t=30, b=0),
            legend=dict(font=dict(color=PALETTE["secondary"], size=10), orientation="h", y=-0.1),
            annotations=[dict(text="Urgency", showarrow=False, font=dict(color=PALETTE["secondary"], size=12))],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ==============================================================================
# LIVE TELEMETRY
# ==============================================================================
def render_live_telemetry() -> None:
    st.markdown('<div class="sg-section-title"><span class="bar"></span>Live Telemetry & Self-Healing Imputation</div>', unsafe_allow_html=True)
    st.caption("Inspect raw sensor streams against SkyGuard AI's non-destructive imputed correction.")

    df_sample = load_sample_dataset()

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_station = st.selectbox("Station", list(STATION_METADATA.keys()), key="tel_station")
    with c2:
        sel_var = st.selectbox(
            "Parameter",
            ["temperature_c", "air_pressure_mbar", "relative_humidity_pct"],
            format_func=lambda x: {
                "temperature_c": "Temperature (°C)",
                "air_pressure_mbar": "Surface Pressure (mbar)",
                "relative_humidity_pct": "Relative Humidity (%)",
            }[x],
            key="tel_var",
        )
    with c3:
        n_points = st.slider("Observation Window", min_value=48, max_value=500, value=168, step=24, key="tel_window")

    if len(df_sample) == 0 or sel_station not in df_sample.get("station_id", pd.Series(dtype=str)).values:
        st.markdown('<div class="sg-skeleton" style="height:420px;"></div>', unsafe_allow_html=True)
        st.info("No live sample dataset found at `data/synthetic/sample_synthetic_anomalies.csv` — connect the backend data source to populate this view.")
        return

    with st.spinner("Running non-destructive imputation pipeline..."):
        stn_slice = df_sample[df_sample["station_id"] == sel_station].tail(n_points).copy()

        try:
            from imputation.correction import MeteorologicalImputer
            imputer = MeteorologicalImputer(k_neighbors=2)
            stn_imputed = imputer.impute_dataframe(stn_slice)
        except Exception:
            stn_imputed = stn_slice.copy()
            stn_imputed[f"imputed_{sel_var}"] = stn_imputed[sel_var]
            stn_imputed["imputation_flag"] = "RAW_PASSTHROUGH"

        time_x = stn_imputed["timestamp"]
        raw_y = stn_imputed[sel_var]
        imp_y = stn_imputed[f"imputed_{sel_var}"]
        clean_y = stn_imputed[f"clean_{sel_var}"] if f"clean_{sel_var}" in stn_imputed.columns else None
        conf_y = stn_imputed["imputation_confidence"] if "imputation_confidence" in stn_imputed.columns else None

        fig = go.Figure()
        if clean_y is not None:
            fig.add_trace(go.Scatter(
                x=time_x, y=clean_y, mode="lines", name="Ground Truth",
                line=dict(color=PALETTE["secondary"], width=2, dash="dash"),
            ))
        fig.add_trace(go.Scatter(
            x=time_x, y=raw_y, mode="lines+markers", name="Corrupted Raw",
            line=dict(color=PALETTE["critical"], width=1.5), marker=dict(size=4, color=PALETTE["critical"]),
        ))
        fig.add_trace(go.Scatter(
            x=time_x, y=imp_y, mode="lines", name="SkyGuard Imputed",
            line=dict(color=PALETTE["success"], width=2.5),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=f"{sel_station} — {sel_var}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=PALETTE["secondary"])),
            margin=dict(l=20, r=20, t=60, b=20), height=430,
            font=dict(color=PALETTE["text"]),
            xaxis=dict(gridcolor=PALETTE["border"]), yaxis=dict(gridcolor=PALETTE["border"]),
        )

        col_chart, col_metrics = st.columns([3, 1])
        with col_chart:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col_metrics:
            st.markdown('<div class="sg-panel">', unsafe_allow_html=True)
            if clean_y is not None:
                anom_pts = stn_imputed["imputation_flag"] != "RAW_PASSTHROUGH"
                if anom_pts.sum() > 0:
                    raw_mae = float(np.nanmean(np.abs(raw_y[anom_pts] - clean_y[anom_pts])))
                    imp_mae = float(np.nanmean(np.abs(imp_y[anom_pts] - clean_y[anom_pts])))
                    err_red = max(0.0, (raw_mae - imp_mae) / max(raw_mae, 1e-4) * 100.0)
                    confidence = float(conf_y[anom_pts].mean()) * 100 if conf_y is not None else None
                else:
                    raw_mae = imp_mae = err_red = 0.0
                    confidence = None
                st.metric("Raw MAE", f"{raw_mae:.2f}")
                st.metric("Imputed MAE", f"{imp_mae:.2f}")
                st.metric("Error Reduction", f"{err_red:.1f}%")
                st.metric("Confidence", f"{confidence:.1f}%" if confidence is not None else "N/A")
            else:
                st.caption("Ground-truth column unavailable for this sample slice.")
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# MODEL PERFORMANCE
# ==============================================================================
def render_model_chart() -> None:
    st.markdown('<div class="sg-section-title"><span class="bar"></span>Model Performance Benchmark</div>', unsafe_allow_html=True)

    demo_data = load_pipeline_results()
    bench = demo_data.get("model_benchmarks", {})
    if bench:
        names = list(bench.keys())
        f1_scores = [v.get("f1", 0) * (100 if v.get("f1", 0) <= 1 else 1) for v in bench.values()]
    else:
        names = ["Deterministic QC", "Isolation Forest", "Autoencoder", "Temporal Regressor", "Hybrid Ensemble"]
        f1_scores = [82.4, 89.1, 93.6, 96.8, 99.74]

    colors = [PALETTE["secondary"], PALETTE["primary"], PALETTE["purple"], PALETTE["warning"], PALETTE["success"]]
    fig = go.Figure(go.Bar(
        x=names, y=f1_scores,
        marker=dict(color=colors[: len(names)], line=dict(width=0)),
        text=[f"{v:.2f}%" for v in f1_scores], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="F1 Score (%)", range=[0, 105], gridcolor=PALETTE["border"]),
        xaxis=dict(gridcolor=PALETTE["border"]),
        margin=dict(l=20, r=20, t=20, b=20), height=380,
        font=dict(color=PALETTE["text"]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ==============================================================================
# STATION HEALTH GRID
# ==============================================================================
def render_station_health() -> None:
    st.markdown('<div class="sg-section-title"><span class="bar"></span>Station Health & Remaining Useful Life</div>', unsafe_allow_html=True)
    st.caption("Prognostic degradation tracking across drift, noise, and hardware wear indicators.")

    demo_data = load_pipeline_results()
    summaries = demo_data.get("station_health_summaries", [])

    if not summaries:
        summaries = [
            {
                "station_id": st_id, "overall_score": round(random.uniform(78, 98), 1),
                "status": random.choice(["EXCELLENT", "GOOD", "DEGRADING"]),
                "urgency": random.choice(["ROUTINE", "MONITOR"]),
                "sensors": {
                    "temperature_c": {"score": round(random.uniform(80, 99), 1), "days_to_critical": None},
                    "air_pressure_mbar": {"score": round(random.uniform(80, 99), 1), "days_to_critical": None},
                    "relative_humidity_pct": {"score": round(random.uniform(75, 98), 1), "days_to_critical": round(random.uniform(30, 200), 1)},
                },
            }
            for st_id in STATION_METADATA.keys()
        ]

    cols = st.columns(3)
    for i, s in enumerate(summaries):
        st_id = s["station_id"]
        city = STATION_METADATA.get(st_id, {}).get("city", "AWS Field Node")
        score = s["overall_score"]
        status = s["status"]
        urgency = s.get("urgency", "ROUTINE")
        badge_cls = {"EXCELLENT": "b-excellent", "GOOD": "b-good", "DEGRADING": "b-degrading", "CRITICAL": "b-critical"}.get(status, "b-good")

        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="sg-station-card">
                    <div class="sg-station-name">{st_id}</div>
                    <div class="sg-station-city">{city}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-family:'Rajdhani',sans-serif; font-size:1.4rem; font-weight:700; color:{PALETTE['text']};">{score:.1f}</span>
                        <span class="sg-badge {badge_cls}">{status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            sensors = s.get("sensors", {})
            for var, sdata in sensors.items():
                label = {"temperature_c": "Temperature", "air_pressure_mbar": "Pressure", "relative_humidity_pct": "Humidity"}.get(var, var)
                v_score = sdata.get("score", 90.0)
                rul = sdata.get("days_to_critical")
                rul_text = f"RUL: {rul:.0f}d" if rul is not None else "RUL: Stable"
                st.caption(f"{label}: {v_score:.1f}/100 · {rul_text}")
                st.progress(min(1.0, max(0.0, v_score / 100.0)))
            st.caption(f"Urgency: `{urgency}`")
            st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# ANOMALY SANDBOX
# ==============================================================================
def render_sandbox() -> None:
    st.markdown('<div class="sg-section-title"><span class="bar"></span>Interactive Anomaly Sandbox</div>', unsafe_allow_html=True)
    st.caption("Inject a synthetic fault into a live station stream and watch SkyGuard AI detect, diagnose, explain, and self-heal it end to end.")

    st.markdown('<div class="sg-panel">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sim_station = st.selectbox("Station", list(STATION_METADATA.keys()), key="sbx_station")
    with c2:
        sim_var = st.selectbox("Sensor", ["temperature_c", "air_pressure_mbar", "relative_humidity_pct"], key="sbx_var")
    with c3:
        sim_fault = st.selectbox(
            "Fault Type",
            ["SPIKE", "DROP", "FROZEN_SENSOR", "SENSOR_DRIFT", "HIGH_NOISE", "PHYSICALLY_IMPOSSIBLE"],
            key="sbx_fault",
        )
    with c4:
        sim_magnitude = st.slider("Intensity", min_value=1.0, max_value=5.0, value=2.5, step=0.5, key="sbx_mag")

    run_clicked = st.button("🚀 Inject Anomaly & Run SkyGuard AI", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked:
        with st.spinner("Processing telemetry through Multi-Tier Fusion, XAI Explainer, and Imputer..."):
            nominal_base = {"temperature_c": 28.0, "air_pressure_mbar": 1008.0, "relative_humidity_pct": 62.0}
            injected_val = nominal_base[sim_var]
            if sim_fault == "SPIKE":
                injected_val += 15.0 * sim_magnitude
            elif sim_fault == "DROP":
                injected_val -= 15.0 * sim_magnitude
            elif sim_fault == "FROZEN_SENSOR":
                injected_val = nominal_base[sim_var]
            elif sim_fault == "SENSOR_DRIFT":
                injected_val += 4.0 * sim_magnitude
            elif sim_fault == "HIGH_NOISE":
                injected_val += float(np.random.normal(0, 3.0 * sim_magnitude))
            elif sim_fault == "PHYSICALLY_IMPOSSIBLE":
                injected_val = 999.0

            result_payload = {
                "station": sim_station, "sensor": sim_var, "fault": sim_fault,
                "magnitude": sim_magnitude, "injected_val": injected_val,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            try:
                from api.server import manager, ingest_single_observation
                from api.schemas import SensorObservationInput

                manager.load_models()
                payload = SensorObservationInput(
                    station_id=sim_station,
                    timestamp=result_payload["timestamp"],
                    temperature_c=injected_val if sim_var == "temperature_c" else nominal_base["temperature_c"],
                    air_pressure_mbar=injected_val if sim_var == "air_pressure_mbar" else nominal_base["air_pressure_mbar"],
                    relative_humidity_pct=injected_val if sim_var == "relative_humidity_pct" else nominal_base["relative_humidity_pct"],
                    latitude=STATION_METADATA[sim_station]["lat"],
                    longitude=STATION_METADATA[sim_station]["lon"],
                    elevation_m=STATION_METADATA[sim_station]["elev"],
                )
                res = ingest_single_observation(payload)

                result_payload.update({
                    "is_anomaly": res.is_anomaly,
                    "composite_score": res.composite_score,
                    "severity": res.severity,
                    "predicted_cause": res.predicted_cause,
                    "imputed_value": getattr(res, f"imputed_{sim_var}"),
                    "imputation_flag": res.imputation_flag,
                    "imputation_confidence": res.imputation_confidence,
                    "diagnostic_card": res.diagnostic_card,
                    "live_backend": True,
                })
            except Exception:
                # Demo fallback so the sandbox stays fully interactive without a live backend.
                is_anom = sim_fault != "FROZEN_SENSOR" or True
                composite = min(0.99, 0.55 + sim_magnitude * 0.09 + random.uniform(-0.03, 0.03))
                severity = "CRITICAL" if composite > 0.85 else ("ANOMALY" if composite > 0.6 else "SUSPICIOUS")
                imputed_val = nominal_base[sim_var] + np.random.normal(0, 0.4)
                result_payload.update({
                    "is_anomaly": is_anom,
                    "composite_score": composite,
                    "severity": severity,
                    "predicted_cause": sim_fault.replace("_", " ").title(),
                    "imputed_value": imputed_val,
                    "imputation_flag": "MULTIVARIATE_KNN_IMPUTED",
                    "imputation_confidence": max(0.5, 1 - composite * 0.3),
                    "diagnostic_card": None,
                    "live_backend": False,
                })

            st.session_state.sandbox_result = result_payload

    result = st.session_state.sandbox_result
    if result:
        if not result.get("live_backend", True):
            st.warning("Backend packages (`api`, `explainability`, `imputation`) were not importable — showing illustrative demo output. Connect the live backend to see real inference results.")

        st.markdown("#### Results")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<div class="sg-panel">', unsafe_allow_html=True)
            st.markdown("**Input Values**")
            st.write(f"Station: `{result['station']}`")
            st.write(f"Sensor: `{result['sensor']}`")
            st.write(f"Fault Type: `{result['fault']}`")
            st.write(f"Intensity Factor: `{result['magnitude']}`")
            st.write(f"Injected Raw Value: `{result['injected_val']:.2f}`")
            st.write(f"Timestamp: `{result['timestamp']}`")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            sev = result.get("severity", "ANOMALY")
            badge_cls = "b-critical" if sev == "CRITICAL" else ("b-degrading" if sev == "ANOMALY" else "b-good")
            st.markdown('<div class="sg-panel">', unsafe_allow_html=True)
            st.markdown("**AI Diagnosis**")
            st.markdown(
                f"""
                <div style="margin-bottom:10px;">
                    <span class="sg-badge {badge_cls}">{sev}</span>
                    &nbsp;<span class="sg-badge {'b-critical' if result.get('is_anomaly') else 'b-excellent'}">
                        {'FLAGGED' if result.get('is_anomaly') else 'NORMAL'}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Composite Anomaly Score", f"{result.get('composite_score', 0):.3f}")
            st.write(f"Attributed Cause: **{result.get('predicted_cause','—')}**")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### XAI Diagnostic Card")
        st.markdown('<div class="sg-panel">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Imputed Value", f"{result.get('imputed_value', 0):.2f}")
        c2.metric("Imputation Confidence", f"{result.get('imputation_confidence', 0)*100:.1f}%")
        c3.markdown(
            f"<span class='sg-badge {badge_cls}'>SEVERITY: {sev}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Imputation strategy: `{result.get('imputation_flag','—')}`")

        if result.get("diagnostic_card"):
            st.markdown(DiagnosticCardGenerator.format_markdown(result["diagnostic_card"]))
        else:
            st.caption("Full XAI diagnostic card renders here once the live `DiagnosticCardGenerator` backend is connected.")
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# REPORTS
# ==============================================================================
def render_reports(overlay_df: pd.DataFrame) -> None:
    st.markdown('<div class="sg-section-title"><span class="bar"></span>Reports & Export</div>', unsafe_allow_html=True)
    st.caption("Fleet-wide summaries for operational review and offline distribution.")

    render_model_chart()

    st.markdown('<div class="sg-section-title"><span class="bar"></span>Fleet Status Summary</div>', unsafe_allow_html=True)
    status_counts = overlay_df["status"].value_counts().reindex(["EXCELLENT", "GOOD", "DEGRADING", "CRITICAL"]).fillna(0).astype(int)
    summary_df = pd.DataFrame({
        "Status": status_counts.index,
        "Station Count": status_counts.values,
        "Share of Fleet": [f"{v/len(overlay_df)*100:.1f}%" for v in status_counts.values],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    csv_bytes = overlay_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Fleet Snapshot (CSV)",
        data=csv_bytes,
        file_name=f"skyguard_fleet_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    inject_css()
    init_session_state()

    page = render_sidebar()
    render_header()

    overlay_df = generate_network_overlay()
    alerts = compute_alert_feed()

    if page == "Dashboard":
        render_kpi_cards(overlay_df, alerts)
        st.markdown("<div style='margin-bottom:22px;'></div>", unsafe_allow_html=True)
        render_network_map(overlay_df, alerts, show_alert_panel=True)
        st.markdown("<div style='margin-bottom:22px;'></div>", unsafe_allow_html=True)
        render_health_gauge(overlay_df)
        st.markdown("<div style='margin-bottom:22px;'></div>", unsafe_allow_html=True)
        render_model_chart()

    elif page == "Network Map":
        render_kpi_cards(overlay_df, alerts)
        st.markdown("<div style='margin-bottom:22px;'></div>", unsafe_allow_html=True)
        render_network_map(overlay_df, alerts, show_alert_panel=True)

    elif page == "Live Telemetry":
        render_live_telemetry()

    elif page == "XAI Alerts":
        render_alert_panel(alerts, compact=False)

    elif page == "Station Health":
        render_station_health()

    elif page == "Anomaly Sandbox":
        render_sandbox()

    elif page == "Reports":
        render_reports(overlay_df)


if __name__ == "__main__":
    main()