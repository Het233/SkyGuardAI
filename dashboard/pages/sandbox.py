"""
SkyGuard AI — Interactive Anomaly Sandbox
dashboard/pages/sandbox.py

Injects synthetic sensor faults into the live SkyGuard pipeline and renders the
full detection → imputation → explanation chain returned by the backend.
"""

import asyncio
import inspect
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Backend imports are deferred into render_sandbox() so that a missing
# module never crashes the whole Streamlit app at import time.


# ──────────────────────────────────────────────────────────────────────────────
# Design tokens
# ──────────────────────────────────────────────────────────────────────────────

CYAN = "#22E6F5"
VIOLET = "#A970FF"
INK = "#070B18"

FAULT_TYPES = [
    "SPIKE",
    "DROP",
    "FROZEN_SENSOR",
    "SENSOR_DRIFT",
    "HIGH_NOISE",
    "PHYSICALLY_IMPOSSIBLE",
]

FAULT_NOTES = {
    "SPIKE": "Sudden upward excursion far outside the sensor's recent envelope.",
    "DROP": "Sudden collapse toward the floor of the sensor's range.",
    "FROZEN_SENSOR": "Reading locks to the last value — no variance at all.",
    "SENSOR_DRIFT": "Slow calibration bias accumulating on top of the true signal.",
    "HIGH_NOISE": "Signal buried in amplified random jitter.",
    "PHYSICALLY_IMPOSSIBLE": "Value pushed outside the physical limits of the instrument.",
}

# sensor → (unit, nominal value, plausible span, hard physical limits)
SENSOR_PROFILES = {
    "temperature": ("°C", 27.0, 12.0, (-90.0, 60.0)),
    "humidity": ("%", 62.0, 25.0, (0.0, 100.0)),
    "pressure": ("hPa", 1008.0, 12.0, (850.0, 1090.0)),
    "wind_speed": ("m/s", 4.2, 5.0, (0.0, 120.0)),
    "wind_direction": ("°", 180.0, 90.0, (0.0, 360.0)),
    "rainfall": ("mm", 1.5, 6.0, (0.0, 500.0)),
    "solar_radiation": ("W/m²", 480.0, 350.0, (0.0, 1400.0)),
}


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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve(value):
    """Run coroutines returned by the async backend on a private event loop."""
    if inspect.isawaitable(value):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(value)
        finally:
            loop.close()
    return value


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
        "id": "station_id",
        "stationid": "station_id",
        "station": "station_id",
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "long": "longitude",
        "alt": "elevation",
        "altitude": "elevation",
        "elevation_m": "elevation",
        "name": "station_name",
    }
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    return df


def _station_row(df: pd.DataFrame, station_id: str) -> dict:
    match = df[df["station_id"].astype(str) == str(station_id)]
    return match.iloc[0].to_dict() if not match.empty else {}


def _pick(obj, *names, default=None):
    """Read the first attribute or key that exists on a backend response."""
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


def _build_payload(fields: dict, schema_cls) -> object:
    """Construct the schema object using only the fields it actually declares."""
    declared = getattr(schema_cls, "model_fields", None)
    if declared is None:
        declared = getattr(schema_cls, "__fields__", {})
    accepted = {k: v for k, v in fields.items() if k in declared}
    return schema_cls(**accepted)


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
                    {"range": [0, 50], "color": "rgba(169,112,255,.16)"},
                    {"range": [50, 80], "color": "rgba(169,112,255,.28)"},
                    {"range": [80, 100], "color": "rgba(34,230,245,.26)"},
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
    return "".join(f"<div class='sg-row'><span>{k}</span><b>{v}</b></div>" for k, v in pairs)


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────

def render_sandbox(station_metadata):
    # Deferred backend imports – errors here must not blank the whole dashboard.
    try:
        from api.server import manager, ingest_single_observation  # noqa: F401
        from api.schemas import SensorObservationInput  # noqa: F401
        _backend_available = True
    except Exception as _import_err:
        _backend_available = False
        _import_err_msg = str(_import_err)

    try:
        from explainability.diagnostic_card import DiagnosticCardGenerator as _DCG
        _dcg_available = True
    except Exception:
        _DCG = None  # type: ignore[assignment]
        _dcg_available = False

    _inject_css()
    st.markdown("<div class='sg-wrap'>", unsafe_allow_html=True)
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

    if not _backend_available:
        st.error(
            "Sandbox unavailable \u2014 backend API could not be imported. "
            f"Ensure `api.server`, `api.schemas`, and `explainability.diagnostic_card` "
            f"are installed and importable from the project root. Error: {_import_err_msg}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    stations = _stations_frame(station_metadata)
    if stations.empty or "station_id" not in stations.columns:
        st.error("No stations available. Load station metadata to use the sandbox.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    left, right = st.columns([1, 1], gap="large")

    # ── Controls ────────────────────────────────────────────────────────────
    with left:
        station_ids = stations["station_id"].astype(str).tolist()

        def _label(sid):
            row = _station_row(stations, sid)
            name = row.get("station_name")
            return f"{sid} — {name}" if isinstance(name, str) and name else str(sid)

        station_id = st.selectbox("Station", station_ids, format_func=_label)

        sensor = st.selectbox(
            "Sensor",
            list(SENSOR_PROFILES.keys()),
            format_func=lambda s: s.replace("_", " ").capitalize(),
        )

        fault_type = st.selectbox("Fault type", FAULT_TYPES, format_func=lambda f: f.replace("_", " ").title())
        st.caption(FAULT_NOTES[fault_type])

        intensity = st.slider("Intensity", 1, 10, 6, help="How far the fault pushes the reading off course.")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        inject = st.button("🚀 Inject anomaly", type="primary")

    # ── Live station context ────────────────────────────────────────────────
    row = _station_row(stations, station_id)
    unit, nominal, span, limits = SENSOR_PROFILES[sensor]
    rng = np.random.default_rng(abs(hash((str(station_id), sensor))) % (2**32))
    baseline = round(float(nominal + rng.normal(0.0, span * 0.12)), 3)
    corrupted = _corrupt(baseline, fault_type, intensity, span, limits, rng)

    with right:
        st.markdown(
            f"""
            <div class="sg-card">
              <h3>Live station</h3>
              {_rows([
                ("Station", row.get("station_name") or station_id),
                ("ID", station_id),
                ("Latitude", _fmt(row.get("latitude"), 4, "°")),
                ("Longitude", _fmt(row.get("longitude"), 4, "°")),
                ("Elevation", _fmt(row.get("elevation"), 1, " m")),
              ])}
            </div>
            <div class="sg-card">
              <h3>Current sensor values</h3>
              {_rows([
                (sensor.replace("_", " ").capitalize(), f"{baseline:,.3f} {unit}"),
                ("Value after fault", f"{corrupted:,.3f} {unit}"),
                ("Fault", fault_type.replace("_", " ").title()),
                ("Intensity", f"{intensity} / 10"),
                ("Valid range", f"{limits[0]:g} – {limits[1]:g} {unit}"),
              ])}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not inject:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Run the pipeline ────────────────────────────────────────────────────
    with st.spinner("Loading models and scoring the observation…"):
        _resolve(manager.load_models())
        payload = _build_payload(
            {
                "station_id": str(station_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sensor": sensor,
                "sensor_name": sensor,
                "sensor_type": sensor,
                "parameter": sensor,
                "variable": sensor,
                "value": corrupted,
                "raw_value": corrupted,
                "observed_value": corrupted,
                "unit": unit,
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "elevation": row.get("elevation"),
                "fault_type": fault_type,
                "injected_fault": fault_type,
                "intensity": float(intensity),
            },
            SensorObservationInput,
        )
        res = _resolve(ingest_single_observation(payload))

    severity = _pick(res, "severity", "severity_level", "anomaly_severity")
    confidence = _as_fraction(_pick(res, "confidence", "confidence_score", "detection_confidence"))
    composite = _pick(res, "composite_score", "composite", "anomaly_score")
    cause = _pick(res, "predicted_cause", "cause", "root_cause", "predicted_fault")

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

    # ── Imputation + gauge ──────────────────────────────────────────────────
    imp = _pick(res, "imputation", "imputation_result", "imputation_details", default=res)
    raw_value = _pick(imp, "raw_value", "original_value", "value", default=corrupted)
    imputed_value = _pick(imp, "imputed_value", "corrected_value", "repaired_value")
    strategy = _pick(imp, "strategy", "imputation_strategy", "method")
    imp_conf = _as_fraction(_pick(imp, "imputation_confidence", "confidence", "confidence_score"))

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
                ("Raw value", f"{_fmt(raw_value)} {unit}"),
                ("Imputed value", f"{_fmt(imputed_value)} {unit}"),
                ("Correction", "—" if delta is None else f"{delta:+,.3f} {unit}"),
                ("Strategy", _fmt(strategy)),
                ("Confidence", "—" if imp_conf is None else f"{imp_conf * 100:.1f}%"),
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

    # ── Diagnostic card ─────────────────────────────────────────────────────
    st.markdown("<div class='sg-xai'>", unsafe_allow_html=True)
    if _dcg_available and _DCG is not None:
        try:
            diag_card = getattr(res, 'diagnostic_card', None) or res
            st.markdown(_DCG.format_markdown(diag_card))
        except Exception as _dcg_err:
            st.info(f"Diagnostic card could not be rendered: {_dcg_err}")
    else:
        st.info("DiagnosticCardGenerator unavailable.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
