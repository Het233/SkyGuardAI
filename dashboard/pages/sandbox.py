"""
SkyGuard AI — Interactive Anomaly Sandbox
dashboard/pages/sandbox.py

Injects synthetic sensor faults into the live SkyGuard pipeline and renders the
full detection → imputation → explanation chain returned by the backend.

Architecture (post-refactor):
    Streamlit Cloud  ──HTTPS──▶  Render FastAPI  (/api/v1/ingest/single)

The Sandbox no longer imports api.server directly.  All backend communication
goes through HTTP using the BACKEND_URL configured in dashboard/config.py.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Resolve config from dashboard root
_DASH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

try:
    from config import BACKEND_URL, HEALTH_ENDPOINT, INGEST_SINGLE
except ImportError:
    BACKEND_URL     = os.getenv("BACKEND_URL", "http://localhost:8000")
    HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
    INGEST_SINGLE   = f"{BACKEND_URL}/api/v1/ingest/single"


# ──────────────────────────────────────────────────────────────────────────────
# Design tokens
# ──────────────────────────────────────────────────────────────────────────────

CYAN   = "#22E6F5"
VIOLET = "#A970FF"
INK    = "#070B18"

FAULT_TYPES = [
    "SPIKE",
    "DROP",
    "FROZEN_SENSOR",
    "SENSOR_DRIFT",
    "HIGH_NOISE",
    "PHYSICALLY_IMPOSSIBLE",
]

FAULT_NOTES = {
    "SPIKE":                "Sudden upward excursion far outside the sensor's recent envelope.",
    "DROP":                 "Sudden collapse toward the floor of the sensor's range.",
    "FROZEN_SENSOR":        "Reading locks to the last value — no variance at all.",
    "SENSOR_DRIFT":         "Slow calibration bias accumulating on top of the true signal.",
    "HIGH_NOISE":           "Signal buried in amplified random jitter.",
    "PHYSICALLY_IMPOSSIBLE":"Value pushed outside the physical limits of the instrument.",
}

# sensor → (unit, nominal value, plausible span, hard physical limits)
SENSOR_PROFILES = {
    "temperature":    ("°C",   27.0,  12.0, (-90.0,  60.0)),
    "humidity":       ("%",    62.0,  25.0, (  0.0, 100.0)),
    "pressure":       ("hPa", 1008.0, 12.0, (850.0, 1090.0)),
    "wind_speed":     ("m/s",  4.2,   5.0,  (  0.0, 120.0)),
    "wind_direction": ("°",   180.0,  90.0, (  0.0, 360.0)),
    "rainfall":       ("mm",   1.5,   6.0,  (  0.0, 500.0)),
    "solar_radiation":("W/m²", 480.0, 350.0,(  0.0,1400.0)),
}

# HTTP timeouts (connect_sec, read_sec)
_HTTP_TIMEOUT = (5, 30)


# ──────────────────────────────────────────────────────────────────────────────
# Backend connectivity
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _probe_backend(url: str) -> tuple[bool, str]:
    """Probe /health. Returns (reachable, detail_string).
    Result is cached for 30 s so the probe does not fire on every widget
    interaction — the TTL still lets it re-check after a connection failure.
    """
    try:
        r = requests.get(url, timeout=_HTTP_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            ver   = data.get("version", "?")
            dev   = data.get("device", "?")
            loaded = data.get("models_loaded", False)
            detail = f"v{ver} · {dev} · models {'✓' if loaded else '✗'}"
            return True, detail
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timed out"
    except Exception as exc:
        return False, str(exc)


def _call_ingest(payload: dict) -> dict | None:
    """POST to /api/v1/ingest/single. Returns response dict or raises RuntimeError."""
    try:
        r = requests.post(INGEST_SINGLE, json=payload, timeout=_HTTP_TIMEOUT)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach backend at **{BACKEND_URL}**.\n\n"
            "Check that the Render service is running and that `BACKEND_URL` is "
            "set correctly in your Streamlit secrets."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Backend timed out after {_HTTP_TIMEOUT[1]} s. "
            "The model may still be loading — try again in a few seconds."
        ) from exc

    if r.status_code == 422:
        raise RuntimeError(
            f"Backend rejected the payload (HTTP 422 — Unprocessable Entity).\n\n"
            f"Detail: `{r.text[:400]}`"
        )
    if r.status_code >= 500:
        raise RuntimeError(
            f"Backend internal error (HTTP {r.status_code}).\n\n"
            f"Detail: `{r.text[:400]}`"
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Backend returned HTTP {r.status_code}: `{r.text[:400]}`")

    try:
        return r.json()
    except Exception as exc:
        raise RuntimeError(
            f"Backend returned a non-JSON response (HTTP {r.status_code}).\n\n"
            f"Body: `{r.text[:200]}`"
        ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .sg-wrap {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            color: #E7ECFF;
        }}
        .sg-hero {{
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 22px;
            background:
                radial-gradient(120% 160% at 8% 0%, rgba(34,230,245,.22), transparent 55%),
                radial-gradient(120% 160% at 92% 100%, rgba(169,112,255,.26), transparent 58%),
                rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.12);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            box-shadow: 0 18px 48px rgba(3,6,20,.55);
        }}
        .sg-hero h1 {{
            margin: 0;
            font-size: clamp(1.6rem, 3.4vw, 2.3rem);
            font-weight: 700;
            letter-spacing: -.02em;
            background: linear-gradient(96deg, {CYAN}, {VIOLET});
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }}
        .sg-hero p {{
            margin: 8px 0 0;
            max-width: 68ch;
            color: rgba(231,236,255,.72);
            font-size: .95rem;
            line-height: 1.6;
        }}
        /* Backend status banner */
        .sg-backend-ok {{
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(43,255,168,.07);
            border: 1px solid rgba(43,255,168,.35);
            border-radius: 10px;
            padding: 7px 14px;
            font-family: 'JetBrains Mono', monospace; font-size: .75rem;
            color: #2bffa8; margin-bottom: 14px;
        }}
        .sg-backend-err {{
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(255,84,112,.07);
            border: 1px solid rgba(255,84,112,.35);
            border-radius: 10px;
            padding: 7px 14px;
            font-family: 'JetBrains Mono', monospace; font-size: .75rem;
            color: #ff5470; margin-bottom: 14px;
        }}
        .sg-dot-ok  {{ width:8px;height:8px;border-radius:50%;background:#2bffa8;box-shadow:0 0 6px #2bffa8;flex-shrink:0; }}
        .sg-dot-err {{ width:8px;height:8px;border-radius:50%;background:#ff5470;box-shadow:0 0 6px #ff5470;flex-shrink:0; }}
        .sg-card {{
            border-radius: 20px;
            padding: 20px 22px;
            background: rgba(255,255,255,.05);
            border: 1px solid rgba(255,255,255,.11);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 12px 34px rgba(3,6,20,.45);
            margin-bottom: 16px;
        }}
        .sg-card h3 {{
            margin: 0 0 14px;
            font-size: .82rem;
            font-weight: 600;
            letter-spacing: .06em;
            color: rgba(231,236,255,.6);
        }}
        .sg-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 16px;
            padding: 9px 0;
            border-bottom: 1px solid rgba(255,255,255,.07);
        }}
        .sg-row:last-child {{ border-bottom: none; }}
        .sg-row span {{ color: rgba(231,236,255,.6); font-size: .86rem; }}
        .sg-row b {{
            font-variant-numeric: tabular-nums;
            font-size: 1rem;
            font-weight: 600;
            color: #F2F6FF;
        }}
        .sg-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px;
            margin: 6px 0 20px;
        }}
        .sg-metric {{
            border-radius: 20px;
            padding: 20px 22px;
            background: rgba(255,255,255,.055);
            border: 1px solid rgba(255,255,255,.12);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            position: relative;
            overflow: hidden;
            animation: sg-rise .5s cubic-bezier(.22,.9,.3,1) both;
        }}
        .sg-metric:nth-child(2) {{ animation-delay: .07s; }}
        .sg-metric:nth-child(3) {{ animation-delay: .14s; }}
        .sg-metric:nth-child(4) {{ animation-delay: .21s; }}
        .sg-metric::after {{
            content: "";
            position: absolute;
            inset: auto 0 0 0;
            height: 2px;
            background: linear-gradient(90deg, {CYAN}, {VIOLET});
            opacity: .85;
        }}
        .sg-metric .k {{
            font-size: .78rem;
            color: rgba(231,236,255,.6);
            margin-bottom: 8px;
        }}
        .sg-metric .v {{
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -.02em;
            font-variant-numeric: tabular-nums;
            color: #FFFFFF;
            line-height: 1.15;
            word-break: break-word;
        }}
        .sg-metric .s {{
            margin-top: 6px;
            font-size: .78rem;
            color: rgba(231,236,255,.5);
        }}
        @keyframes sg-rise {{
            from {{ opacity: 0; transform: translateY(14px); }}
            to   {{ opacity: 1; transform: none; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .sg-metric {{ animation: none; }}
        }}
        .sg-xai {{
            border-radius: 20px;
            padding: 8px 26px 18px;
            background: rgba(255,255,255,.04);
            border: 1px solid rgba(255,255,255,.1);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }}
        .sg-xai h1, .sg-xai h2, .sg-xai h3 {{ color: {CYAN}; }}
        div[data-testid="stButton"] > button {{
            width: 100%;
            border-radius: 20px;
            padding: 16px 20px;
            font-size: 1.02rem;
            font-weight: 700;
            letter-spacing: .01em;
            color: {INK};
            border: none;
            background: linear-gradient(96deg, {CYAN}, {VIOLET});
            box-shadow: 0 0 22px rgba(34,230,245,.42), 0 0 46px rgba(169,112,255,.3);
            transition: box-shadow .2s ease, transform .2s ease;
        }}
        div[data-testid="stButton"] > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 0 30px rgba(34,230,245,.6), 0 0 64px rgba(169,112,255,.45);
        }}
        div[data-testid="stButton"] > button:focus-visible {{
            outline: 2px solid #FFFFFF;
            outline-offset: 3px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (unchanged from original — _pick works on plain dicts from JSON)
# ──────────────────────────────────────────────────────────────────────────────

def _stations_frame(station_metadata) -> pd.DataFrame:
    """Normalise whatever the caller passes into a station dataframe."""
    if isinstance(station_metadata, pd.DataFrame):
        df = station_metadata.copy()
    elif isinstance(station_metadata, dict):
        df = pd.DataFrame.from_dict(station_metadata, orient="index")
        df = df.reset_index().rename(columns={"index": "station_id"})
    else:
        df = pd.DataFrame(list(station_metadata))

    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "id": "station_id", "stationid": "station_id", "station": "station_id",
        "lat": "latitude", "lon": "longitude", "lng": "longitude", "long": "longitude",
        "alt": "elevation", "altitude": "elevation", "elevation_m": "elevation",
        "name": "station_name",
    }
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    return df


def _station_row(df: pd.DataFrame, station_id: str) -> dict:
    match = df[df["station_id"].astype(str) == str(station_id)]
    return match.iloc[0].to_dict() if not match.empty else {}


def _pick(obj, *names, default=None):
    """Read the first key that exists in a dict (or attribute on an object)."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _fmt(value, digits=3, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
        return f"{float(value):,.{digits}f}{suffix}"
    return str(value)


def _as_fraction(value):
    """Confidence may arrive as 0–1 or 0–100; normalise to 0–1."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v / 100.0 if v > 1.0 else v


def _corrupt(baseline, fault, intensity, span, limits, rng):
    """Apply the selected fault to a clean baseline reading."""
    low, high = limits
    k = intensity / 5.0  # slider 1–10 → 0.2–2.0

    if fault == "SPIKE":
        value = baseline + span * 3.0 * k
    elif fault == "DROP":
        value = baseline - span * 3.0 * k
    elif fault == "FROZEN_SENSOR":
        value = baseline
    elif fault == "SENSOR_DRIFT":
        value = baseline + span * 1.2 * k
    elif fault == "HIGH_NOISE":
        value = baseline + rng.normal(0.0, span * 1.5 * k)
    elif fault == "PHYSICALLY_IMPOSSIBLE":
        value = high + abs(high - low) * (0.5 + k)
    else:
        value = baseline

    if fault != "PHYSICALLY_IMPOSSIBLE":
        value = float(np.clip(value, low, high))
    return round(float(value), 4)


def _confidence_gauge(fraction: float) -> go.Figure:
    pct = round(fraction * 100.0, 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 40, "color": "#FFFFFF"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "rgba(231,236,255,.45)",
                    "tickfont": {"color": "rgba(231,236,255,.6)", "size": 11},
                },
                "bar": {"color": CYAN, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  50], "color": "rgba(169,112,255,.16)"},
                    {"range": [50, 80], "color": "rgba(169,112,255,.28)"},
                    {"range": [80,100], "color": "rgba(34,230,245,.26)"},
                ],
                "threshold": {
                    "line": {"color": VIOLET, "width": 4},
                    "thickness": 0.82,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=24, r=24, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#E7ECFF"},
    )
    return fig


def _rows(pairs) -> str:
    return "".join(
        f"<div class='sg-row'><span>{k}</span><b>{v}</b></div>"
        for k, v in pairs
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page entry point
# ──────────────────────────────────────────────────────────────────────────────

def render_sandbox(station_metadata):
    _inject_css()
    st.markdown("<div class='sg-wrap'>", unsafe_allow_html=True)

    # ── Hero banner ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sg-hero">
          <h1>Anomaly Sandbox</h1>
          <p>Break a sensor on purpose. Pick a station, choose how it fails, and send the
             reading through the live SkyGuard pipeline to see what the detector flags,
             how the gap is filled, and why.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Backend status indicator ─────────────────────────────────────────────
    reachable, detail = _probe_backend(HEALTH_ENDPOINT)

    # Mask the URL to hide any embedded tokens — show only host
    try:
        from urllib.parse import urlparse
        _host = urlparse(BACKEND_URL).netloc or BACKEND_URL
    except Exception:
        _host = BACKEND_URL

    if reachable:
        st.markdown(
            f"<div class='sg-backend-ok'>"
            f"<span class='sg-dot-ok'></span>"
            f"Backend connected &nbsp;·&nbsp; <code>{_host}</code> &nbsp;·&nbsp; {detail}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='sg-backend-err'>"
            f"<span class='sg-dot-err'></span>"
            f"Backend unavailable &nbsp;·&nbsp; <code>{_host}</code> &nbsp;·&nbsp; {detail}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.error(
            "**Sandbox unavailable** — cannot reach the SkyGuard FastAPI backend.\n\n"
            f"**Backend URL:** `{_host}`\n\n"
            "**For local dev:** run `uvicorn api.server:app --reload --host 0.0.0.0 --port 8000`  \n"
            "**For cloud:** set `BACKEND_URL` in Streamlit secrets to your Render service URL."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Station / fault controls ─────────────────────────────────────────────
    stations = _stations_frame(station_metadata)
    if stations.empty or "station_id" not in stations.columns:
        st.error("No stations available. Load station metadata to use the sandbox.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        station_ids = stations["station_id"].astype(str).tolist()

        def _label(sid):
            row = _station_row(stations, sid)
            name = row.get("station_name")
            return f"{sid} — {name}" if isinstance(name, str) and name else str(sid)

        station_id = st.selectbox("Station", station_ids, format_func=_label)
        sensor     = st.selectbox(
            "Sensor", list(SENSOR_PROFILES.keys()),
            format_func=lambda s: s.replace("_", " ").capitalize(),
        )
        fault_type = st.selectbox(
            "Fault type", FAULT_TYPES,
            format_func=lambda f: f.replace("_", " ").title(),
        )
        st.caption(FAULT_NOTES[fault_type])
        intensity = st.slider("Intensity", 1, 10, 6,
                              help="How far the fault pushes the reading off course.")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        inject = st.button("🚀 Inject anomaly", type="primary")

    # ── Live station context ─────────────────────────────────────────────────
    row = _station_row(stations, station_id)
    unit, nominal, span, limits = SENSOR_PROFILES[sensor]
    rng      = np.random.default_rng(abs(hash((str(station_id), sensor))) % (2**32))
    baseline = round(float(nominal + rng.normal(0.0, span * 0.12)), 3)
    corrupted = _corrupt(baseline, fault_type, intensity, span, limits, rng)

    with right:
        st.markdown(
            f"""
            <div class="sg-card">
              <h3>Live station</h3>
              {_rows([
                ("Station",   row.get("station_name") or station_id),
                ("ID",        station_id),
                ("Latitude",  _fmt(row.get("latitude"),  4, "°")),
                ("Longitude", _fmt(row.get("longitude"), 4, "°")),
                ("Elevation", _fmt(row.get("elevation"), 1, " m")),
              ])}
            </div>
            <div class="sg-card">
              <h3>Current sensor values</h3>
              {_rows([
                (sensor.replace("_", " ").capitalize(), f"{baseline:,.3f} {unit}"),
                ("Value after fault",  f"{corrupted:,.3f} {unit}"),
                ("Fault",             fault_type.replace("_", " ").title()),
                ("Intensity",         f"{intensity} / 10"),
                ("Valid range",       f"{limits[0]:g} – {limits[1]:g} {unit}"),
              ])}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not inject:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Call FastAPI via HTTP ────────────────────────────────────────────────
    # Map the sensor field to the three fields the SensorObservationInput schema
    # actually declares (temperature_c, air_pressure_mbar, relative_humidity_pct).
    # Any sensor that doesn't map to those three is sent with all three as None;
    # the QC layer will flag missing values appropriately.
    _SENSOR_TO_FIELD = {
        "temperature":  "temperature_c",
        "humidity":     "relative_humidity_pct",
        "pressure":     "air_pressure_mbar",
    }
    api_payload: dict = {
        "station_id":              str(station_id),
        "timestamp":               datetime.now(timezone.utc).isoformat(),
        "temperature_c":           None,
        "air_pressure_mbar":       None,
        "relative_humidity_pct":   None,
        "latitude":                row.get("latitude"),
        "longitude":               row.get("longitude"),
        "elevation_m":             row.get("elevation"),
    }
    field = _SENSOR_TO_FIELD.get(sensor)
    if field:
        api_payload[field] = corrupted

    with st.spinner("Sending observation to backend and scoring…"):
        try:
            res = _call_ingest(api_payload)
        except RuntimeError as exc:
            st.error(str(exc))
            st.markdown("</div>", unsafe_allow_html=True)
            return

    # ── Results — res is a plain dict from JSON ──────────────────────────────
    severity   = _pick(res, "severity", "severity_level", "anomaly_severity")
    confidence = _as_fraction(_pick(res, "cause_confidence", "confidence", "confidence_score"))
    composite  = _pick(res, "composite_score", "composite", "anomaly_score")
    cause      = _pick(res, "predicted_cause", "cause", "root_cause", "predicted_fault")

    st.markdown(
        f"""
        <div class="sg-metrics">
          <div class="sg-metric"><div class="k">Severity</div>
            <div class="v">{_fmt(severity, 2)}</div>
            <div class="s">Assigned by the detector</div></div>
          <div class="sg-metric"><div class="k">Confidence</div>
            <div class="v">{'—' if confidence is None else f'{confidence * 100:.1f}%'}</div>
            <div class="s">Certainty in this call</div></div>
          <div class="sg-metric"><div class="k">Composite score</div>
            <div class="v">{_fmt(composite, 3)}</div>
            <div class="s">Blended ensemble output</div></div>
          <div class="sg-metric"><div class="k">Predicted cause</div>
            <div class="v">{_fmt(cause)}</div>
            <div class="s">Injected: {fault_type.replace('_', ' ').title()}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Imputation comparison + gauge ────────────────────────────────────────
    # The /ingest/single response includes top-level imputation fields.
    raw_value     = _pick(res, "raw_value", "original_value", "value", default=corrupted)
    imputed_value = _pick(
        res,
        f"imputed_{_SENSOR_TO_FIELD.get(sensor, 'temperature_c')}",
        "imputed_value", "corrected_value", "repaired_value",
    )
    strategy  = _pick(res, "imputation_flag", "strategy", "imputation_strategy", "method")
    imp_conf  = _as_fraction(_pick(res, "imputation_confidence", "confidence_score"))

    delta = None
    if isinstance(raw_value, (int, float)) and isinstance(imputed_value, (int, float)):
        delta = float(imputed_value) - float(raw_value)

    col_a, col_b = st.columns([1.15, 1], gap="large")
    with col_a:
        st.markdown(
            f"""
            <div class="sg-card">
              <h3>Imputation comparison</h3>
              {_rows([
                ("Raw value",     f"{_fmt(raw_value)} {unit}"),
                ("Imputed value", f"{_fmt(imputed_value)} {unit}"),
                ("Correction",    "—" if delta is None else f"{delta:+,.3f} {unit}"),
                ("Strategy",      _fmt(strategy)),
                ("Confidence",    "—" if imp_conf is None else f"{imp_conf * 100:.1f}%"),
              ])}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown("<div class='sg-card'><h3>Detection confidence</h3></div>", unsafe_allow_html=True)
        st.plotly_chart(
            _confidence_gauge(confidence if confidence is not None else 0.0),
            width="stretch",
            config={"displayModeBar": False},
        )

    # ── Diagnostic card from response JSON ───────────────────────────────────
    st.markdown("<div class='sg-xai'>", unsafe_allow_html=True)
    diag_card = _pick(res, "diagnostic_card")
    if diag_card and isinstance(diag_card, dict):
        # Render the card as structured markdown from the JSON fields
        title   = diag_card.get("title", "Diagnostic Report")
        summary = diag_card.get("summary", "")
        sop     = diag_card.get("sop", "")
        evidence= diag_card.get("evidence", [])
        st.markdown(f"### {title}")
        if summary:
            st.markdown(summary)
        if evidence:
            st.markdown("**Physical evidence checklist**")
            for ev in evidence:
                st.markdown(f"- {ev}")
        if sop:
            st.markdown("**Maintenance SOP**")
            st.markdown(sop)
    else:
        # Try the DiagnosticCardGenerator formatter if available locally
        try:
            from explainability.diagnostic_card import DiagnosticCardGenerator as _DCG
            md = _DCG.format_markdown(res)
            st.markdown(md)
        except Exception:
            st.info(
                "No diagnostic card was returned for this observation. "
                "The detector did not classify this reading as an anomaly, "
                "or the card generator is unavailable in this environment."
            )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
