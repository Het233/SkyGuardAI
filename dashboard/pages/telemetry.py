"""
================================================================================
dashboard/pages/telemetry.py -- SkyGuard AI Telemetry Dashboard
================================================================================
Renders the real AWS telemetry inspector backed by the actual
aws_weather_data_all_india_2023_present.csv dataset (2023-01-01 to 2026-09-11,
826 stations, ~25,000 hourly observations per station).

Data source:   data_loader.get_telemetry_for_station()  (chunked CSV reader)
Station list:  data_loader.get_station_ids()             (parquet, instant)
Summary table: data_loader.get_latest_telemetry_all()    (EDA parquet, 826 rows)

Actual dataset schema (from the CSV):
    station_id, station_name, state, district,
    latitude, longitude, elevation_m, timestamp,
    temperature_c, air_pressure_mbar, relative_humidity_pct

Nothing is fabricated. If data is unavailable, a clear message is shown.
The page explicitly labels this HISTORICAL DATA (not LIVE).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Resolve data_loader from dashboard root
_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

from data_loader import (
    get_station_ids,
    get_telemetry_for_station,
    get_latest_telemetry_all,
)

# ---------------------------------------------------------------------------
# Column mapping: actual CSV columns → display labels & units
# ---------------------------------------------------------------------------
_SENSOR_COLS = {
    "temperature_c":       ("Temperature",       "°C",   "#22e8ff"),
    "air_pressure_mbar":   ("Pressure",          "mbar", "#7c8cff"),
    "relative_humidity_pct": ("Humidity",        "%",    "#2bffa8"),
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.06) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'. Falls back to the original
    color string if parsing fails."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return hex_color

# CSV is present if the big file exists
_CSV_PATH = os.path.join(os.path.dirname(_DASHBOARD_DIR), "aws_weather_data_all_india_2023_present.csv")
_HAS_CSV  = os.path.exists(_CSV_PATH)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def _inject_css() -> None:
    st.markdown("""
<style>
.sg-tele-title {
    font-family:'Orbitron',sans-serif;
    font-size:1.25rem; font-weight:800;
    color:#e7f6ff; letter-spacing:.04em; margin-bottom:2px;
}
.sg-tele-sub {
    font-family:'JetBrains Mono',monospace;
    font-size:.68rem; color:#607898; letter-spacing:.06em;
    text-transform:uppercase; margin-bottom:14px;
}
.sg-tele-badge {
    display:inline-block; padding:3px 12px; border-radius:999px;
    font-family:'JetBrains Mono',monospace; font-size:.62rem;
    font-weight:700; letter-spacing:.1em;
}
.badge-hist {
    background:rgba(124,140,255,.12); border:1px solid rgba(124,140,255,.4);
    color:#7c8cff;
}
.badge-local {
    background:rgba(43,255,168,.09); border:1px solid rgba(43,255,168,.35);
    color:#2bffa8;
}
.badge-warn {
    background:rgba(255,84,112,.10); border:1px solid rgba(255,84,112,.35);
    color:#ff5470;
}
.sg-tele-stat {
    background:#0B1220; border:1px solid rgba(255,255,255,.08);
    border-radius:10px; padding:12px 16px; margin-bottom:8px;
}
.sg-tele-stat-label {
    font-family:'JetBrains Mono',monospace; font-size:.57rem;
    letter-spacing:.08em; text-transform:uppercase; color:#607898;
}
.sg-tele-stat-val {
    font-family:'Orbitron',sans-serif; font-size:1.1rem;
    font-weight:800; color:#deeeff; margin-top:2px;
}
.sg-tele-stat-sub {
    font-family:'Rajdhani',sans-serif; font-size:.68rem;
    color:#3c4e64; margin-top:1px;
}
.sg-tele-metric {
    background:#0B1220; border:1px solid rgba(255,255,255,.08);
    border-left:3px solid var(--tc,#22e8ff);
    border-radius:10px; padding:12px 14px;
}
.sg-tele-metric-label {
    font-family:'JetBrains Mono',monospace; font-size:.58rem;
    letter-spacing:.08em; text-transform:uppercase; color:#607898;
}
.sg-tele-metric-value {
    font-family:'Orbitron',sans-serif; font-weight:800; font-size:1.4rem;
    color:#e8f4ff; line-height:1.1; margin-top:4px;
}
.sg-tele-metric-unit {
    font-size:.78rem; font-weight:500; color:#607898; margin-left:3px;
}
.sg-tele-metric-sub {
    font-family:'Rajdhani',sans-serif; font-size:.68rem; color:#3c4e64;
    margin-top:2px;
}
.sg-tele-loading {
    text-align:center; padding:32px 10px; color:#607898;
    font-family:'JetBrains Mono',monospace; font-size:.8rem; letter-spacing:.04em;
}
.sg-tele-section {
    font-family:'Orbitron',sans-serif; font-size:.80rem; font-weight:700;
    color:#7df9ff; letter-spacing:.10em; text-transform:uppercase;
    margin:18px 0 8px 0; border-bottom:1px solid rgba(34,200,255,.15);
    padding-bottom:4px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stat_card(col, label: str, value: str, sub: str = "") -> None:
    with col:
        st.markdown(
            f'<div class="sg-tele-stat">'
            f'<div class="sg-tele-stat-label">{label}</div>'
            f'<div class="sg-tele-stat-val">{value}</div>'
            f'<div class="sg-tele-stat-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _metric_card(col, label: str, value: str, unit: str, sub: str, color: str) -> None:
    with col:
        st.markdown(
            f'<div class="sg-tele-metric" style="--tc:{color};">'
            f'<div class="sg-tele-metric-label">{label}</div>'
            f'<div class="sg-tele-metric-value">{value}'
            f'<span class="sg-tele-metric-unit">{unit}</span></div>'
            f'<div class="sg-tele-metric-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _status_badge(status: str) -> str:
    colors = {
        "EXCELLENT": ("rgba(43,255,168,.12)", "rgba(43,255,168,.45)", "#2bffa8"),
        "GOOD":      ("rgba(34,200,255,.10)", "rgba(34,200,255,.40)", "#22e8ff"),
        "DEGRADING": ("rgba(255,184,0,.10)",  "rgba(255,184,0,.40)",  "#ffb800"),
        "CRITICAL":  ("rgba(255,84,112,.12)", "rgba(255,84,112,.45)", "#ff5470"),
    }
    bg, bd, fg = colors.get(status, ("rgba(96,120,152,.08)", "rgba(96,120,152,.3)", "#607898"))
    return (
        f'<span style="background:{bg};border:1px solid {bd};color:{fg};'
        f'padding:2px 9px;border-radius:999px;font-size:.65rem;'
        f'font-family:\'JetBrains Mono\',monospace;font-weight:700;'
        f'letter-spacing:.07em;">{status}</span>'
    )


_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c7d6e6", family="Rajdhani, sans-serif"),
    margin=dict(l=10, r=10, t=44, b=10),
    height=380,
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(color="#c7d6e6"),
    ),
    hoverlabel=dict(
        bgcolor="rgba(5,11,20,.95)",
        bordercolor="rgba(34,200,255,.4)",
        font=dict(color="#e7f6ff"),
    ),
)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _render_status_bar(station_ids: list) -> None:
    """Top status banner: dataset info + data-source badge."""
    data_source = "● LOCAL DATASET" if _HAS_CSV else "⚠ PARQUET SUMMARY ONLY"
    badge_cls   = "badge-local" if _HAS_CSV else "badge-warn"
    hist_label  = "HISTORICAL DATA · 2023-01-01 → 2026-09-11 · HOURLY"

    col_badge, col_rest = st.columns([1, 4])
    with col_badge:
        st.markdown(
            f'<div class="sg-tele-badge badge-hist" style="margin-top:4px;">{hist_label}</div>',
            unsafe_allow_html=True,
        )
    with col_rest:
        st.markdown(
            f'<div class="sg-tele-badge {badge_cls}" style="margin-top:4px;">{data_source}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sg-tele-section">TELEMETRY OVERVIEW</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    _stat_card(c1, "AWS Stations",    str(len(station_ids)), "In network")
    _stat_card(c2, "Dataset Period",  "3.75 yrs", "2023-01-01 → 2026-09-11")
    _stat_card(c3, "Obs per Station", "~25,000",  "Hourly readings")
    _stat_card(c4, "Sensors",         "3 active", "Temp · Pressure · Humidity")


def _render_controls(station_ids: list) -> tuple:
    """Station + sensor + date-range controls. Returns (station_id, sensor_col, date_start, date_end)."""
    st.markdown('<div class="sg-tele-section">CONTROLS</div>', unsafe_allow_html=True)

    c_stn, c_sens, c_start, c_end = st.columns([2, 1.5, 1.2, 1.2])

    with c_stn:
        if not station_ids:
            st.error("No station IDs available.")
            return None, None, None, None
        sel_station = st.selectbox(
            "AWS Station",
            station_ids,
            key="sg_tele_station_v2",
        )

    with c_sens:
        sens_options = list(_SENSOR_COLS.keys())
        sel_sensor = st.selectbox(
            "Sensor",
            sens_options,
            format_func=lambda c: _SENSOR_COLS[c][0],
            key="sg_tele_sensor_v2",
        )

    # Default date range: last 30 days of data available
    dataset_end   = datetime(2026, 9, 11)
    dataset_start = datetime(2023, 1, 1)
    default_start = dataset_end - timedelta(days=30)

    with c_start:
        date_start = st.date_input(
            "From",
            value=default_start.date(),
            min_value=dataset_start.date(),
            max_value=dataset_end.date(),
            key="sg_tele_start_v2",
        )
    with c_end:
        date_end = st.date_input(
            "To",
            value=dataset_end.date(),
            min_value=dataset_start.date(),
            max_value=dataset_end.date(),
            key="sg_tele_end_v2",
        )

    return sel_station, sel_sensor, pd.Timestamp(date_start), pd.Timestamp(date_end)


def _load_station_data(station_id: str, date_start: pd.Timestamp, date_end: pd.Timestamp) -> pd.DataFrame:
    """Load + filter station time-series. Shows a spinner while reading the CSV."""
    if not _HAS_CSV:
        return pd.DataFrame()

    with st.spinner(f"Loading {station_id} from dataset (first load ~15–30 s, cached thereafter)…"):
        df = get_telemetry_for_station(station_id)

    if df.empty:
        return df

    mask = (df["timestamp"] >= date_start) & (df["timestamp"] <= date_end + timedelta(days=1))
    return df[mask].reset_index(drop=True)


def _render_sensor_kpis(df: pd.DataFrame, sensor_col: str, station_id: str) -> None:
    """KPI cards for current/min/avg/max of the selected sensor."""
    st.markdown('<div class="sg-tele-section">SENSOR STATISTICS</div>', unsafe_allow_html=True)

    label, unit, color = _SENSOR_COLS[sensor_col]

    cols = [c for c in _SENSOR_COLS if c in df.columns]
    n_cols = len(cols)
    kpi_cols = st.columns(n_cols * 4 if n_cols > 0 else 4)

    for i, col_name in enumerate(cols):
        lbl, unt, clr = _SENSOR_COLS[col_name]
        series = df[col_name].dropna()
        if series.empty:
            _metric_card(kpi_cols[i*4],   f"{lbl} — Current", "N/A",  unt, "No data", clr)
            _metric_card(kpi_cols[i*4+1], f"{lbl} — Min",     "N/A",  unt, "", clr)
            _metric_card(kpi_cols[i*4+2], f"{lbl} — Avg",     "N/A",  unt, "", clr)
            _metric_card(kpi_cols[i*4+3], f"{lbl} — Max",     "N/A",  unt, "", clr)
        else:
            current = float(series.iloc[-1])
            vmin    = float(series.min())
            vmean   = float(series.mean())
            vmax    = float(series.max())
            fmt     = ".1f"
            _metric_card(kpi_cols[i*4],   f"{lbl} · Latest", f"{current:{fmt}}", unt, "Most recent",          clr)
            _metric_card(kpi_cols[i*4+1], f"{lbl} · Min",    f"{vmin:{fmt}}",    unt, "In selected period",   clr)
            _metric_card(kpi_cols[i*4+2], f"{lbl} · Mean",   f"{vmean:{fmt}}",   unt, "In selected period",   clr)
            _metric_card(kpi_cols[i*4+3], f"{lbl} · Max",    f"{vmax:{fmt}}",    unt, "In selected period",   clr)


def _render_trend_chart(df: pd.DataFrame, sensor_col: str, station_id: str) -> None:
    """Interactive time-series chart for the selected sensor."""
    st.markdown('<div class="sg-tele-section">SENSOR TREND</div>', unsafe_allow_html=True)

    if sensor_col not in df.columns or df[sensor_col].dropna().empty:
        st.markdown(
            '<div class="sg-tele-loading">No data for this sensor in the selected period.</div>',
            unsafe_allow_html=True,
        )
        return

    label, unit, color = _SENSOR_COLS[sensor_col]

    # Downsample for chart performance: show at most 5000 points
    plot_df = df[["timestamp", sensor_col]].dropna()
    if len(plot_df) > 5000:
        step = max(1, len(plot_df) // 5000)
        plot_df = plot_df.iloc[::step].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df[sensor_col],
        mode="lines",
        name=f"{label} ({unit})",
        line=dict(color=color, width=1.8),
        hovertemplate=f"<b>%{{x|%Y-%m-%d %H:%M}}</b><br>{label}: %{{y:.2f}} {unit}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"{station_id} — {label}",
            font=dict(color="#e7f6ff", family="Orbitron, sans-serif", size=14),
        ),
        yaxis_title=f"{label} ({unit})",
        xaxis_title="Timestamp",
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="sg_tele_trend")


def _render_multi_sensor(df: pd.DataFrame, station_id: str) -> None:
    """Multi-sensor view: each sensor on its own subplot."""
    st.markdown('<div class="sg-tele-section">MULTI-SENSOR VIEW</div>', unsafe_allow_html=True)

    available = [c for c in _SENSOR_COLS if c in df.columns and not df[c].dropna().empty]
    if not available:
        st.markdown(
            '<div class="sg-tele-loading">No sensor data available for this period.</div>',
            unsafe_allow_html=True,
        )
        return

    for col_name in available:
        label, unit, color = _SENSOR_COLS[col_name]
        plot_df = df[["timestamp", col_name]].dropna()
        if len(plot_df) > 3000:
            step = max(1, len(plot_df) // 3000)
            plot_df = plot_df.iloc[::step].reset_index(drop=True)

        fill_color = _hex_to_rgba(color, alpha=0.06) if color.startswith("#") else color
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df[col_name],
            mode="lines",
            name=label,
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=color, width=1.6),
        ))
        fig.update_layout(
            title=dict(text=f"{label} ({unit})", font=dict(color=color, size=13,
                       family="Orbitron,sans-serif")),
            yaxis_title=f"{label} ({unit})",
            **{**_CHART_LAYOUT, "height": 260},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                        key=f"sg_tele_multi_{col_name}")


def _render_station_table() -> None:
    """Live station table: latest reading for every AWS station (826 rows)."""
    st.markdown('<div class="sg-tele-section">ALL STATIONS — LATEST READING</div>', unsafe_allow_html=True)

    df = get_latest_telemetry_all()
    if df is None or df.empty:
        st.markdown(
            '<div class="sg-tele-loading">Station summary unavailable.</div>',
            unsafe_allow_html=True,
        )
        return

    # Build display dataframe with available columns
    display_cols = {}
    if "station_id"    in df.columns: display_cols["station_id"]    = "Station ID"
    if "station_name"  in df.columns: display_cols["station_name"]  = "Name"
    if "state"         in df.columns: display_cols["state"]         = "State"
    if "temperature"   in df.columns: display_cols["temperature"]   = "Temp (°C)"
    if "pressure"      in df.columns: display_cols["pressure"]      = "Pressure (mbar)"
    if "humidity"      in df.columns: display_cols["humidity"]      = "Humidity (%)"
    if "health_score"  in df.columns: display_cols["health_score"]  = "Health"
    if "status"        in df.columns: display_cols["status"]        = "Status"

    show_df = df[list(display_cols.keys())].rename(columns=display_cols)

    # Search filter
    search = st.text_input(
        "Search by Station ID or State",
        placeholder="e.g. IMD_AWS_0001 or Maharashtra",
        key="sg_tele_table_search",
    )
    if search:
        mask = show_df.apply(
            lambda col: col.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        show_df = show_df[mask]

    st.caption(f"Showing {len(show_df)} of 826 stations · Last reading per station")
    st.dataframe(show_df, use_container_width=True, height=340, hide_index=True)


# ---------------------------------------------------------------------------
# Public entry-point (called by app.py with no arguments)
# ---------------------------------------------------------------------------
def render_live_telemetry() -> None:
    """Render the SkyGuard AI Telemetry Dashboard.

    Loads real data from the 2.6 GB AWS weather CSV via data_loader
    (chunked, cached per station). The page is fully self-contained —
    app.py does not need to pass any data.
    """
    _inject_css()

    # Title
    st.markdown('<div class="sg-tele-title">📈 AWS Telemetry Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sg-tele-sub">IMD · AUTOMATIC WEATHER STATION NETWORK · '
        'HISTORICAL DATA · TEMPERATURE · PRESSURE · HUMIDITY</div>',
        unsafe_allow_html=True,
    )

    # Get station list (fast — parquet)
    station_ids = get_station_ids()
    if not station_ids:
        st.error("No station IDs could be loaded. Check data_loader configuration.")
        return

    # Status bar
    _render_status_bar(station_ids)

    # Controls
    sel_station, sel_sensor, date_start, date_end = _render_controls(station_ids)
    if sel_station is None:
        return

    # Validate date range
    if date_start > date_end:
        st.warning("'From' date must be before 'To' date.")
        return

    # Refresh button
    col_ref, _ = st.columns([1, 5])
    with col_ref:
        if st.button("🔄 Refresh Data", key="sg_tele_refresh"):
            # Clear the per-station cache for this station
            get_telemetry_for_station.clear()
            st.rerun()

    # Load station time-series
    if not _HAS_CSV:
        st.warning(
            "The full telemetry CSV (`aws_weather_data_all_india_2023_present.csv`) "
            "was not found. The station table uses summary data only. "
            "Time-series charts are unavailable."
        )
        _render_station_table()
        return

    df = _load_station_data(sel_station, date_start, date_end)

    if df.empty:
        st.markdown(
            f'<div class="sg-tele-loading">'
            f'No observations found for <b>{sel_station}</b> '
            f'between {date_start.date()} and {date_end.date()}.'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_station_table()
        return

    row_count = len(df)
    ts_min    = df["timestamp"].min()
    ts_max    = df["timestamp"].max()

    # Dataset info row
    c1, c2, c3 = st.columns(3)
    _stat_card(c1, "Records Loaded",   f"{row_count:,}", f"{sel_station}")
    _stat_card(c2, "Period Start",     ts_min.strftime("%Y-%m-%d %H:%M"), "Earliest observation")
    _stat_card(c3, "Period End",       ts_max.strftime("%Y-%m-%d %H:%M"), "Latest observation")

    # Sensor KPIs
    _render_sensor_kpis(df, sel_sensor, sel_station)

    # Trend chart
    _render_trend_chart(df, sel_sensor, sel_station)

    # Multi-sensor
    with st.expander("📊 Multi-Sensor View", expanded=False):
        _render_multi_sensor(df, sel_station)

    # All-stations table
    with st.expander("📋 All Stations — Latest Reading (826 stations)", expanded=False):
        _render_station_table()