"""
dashboard/pages/simulation.py  —  SkyGuard AI Simulation Lab
=============================================================
3-year deterministic telemetry simulation for any of the 826 IMD AWS stations.

Layout
------
  Top row   : controls (station | dates | speed | play/pause/reset)
  Left/main : 4 KPI cards + 4 mini-metrics + 4 Plotly charts
  Right     : fault injection panel + AI detection card

Entry-point called from app.py:
    render_simulation()
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── path resolution ─────────────────────────────────────────────────────────
_DASH = os.path.dirname(os.path.dirname(__file__))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

from data_loader import load_aws_dataset
from simulation.weather_generator import generate_station_series
from simulation.anomaly_engine     import FAULT_TYPES, inject_fault
from simulation.simulator          import get_state, reset_state

# ── constants ────────────────────────────────────────────────────────────────
_CHART_CFG   = {"displayModeBar": False}
_WINDOW      = 336          # hours shown in charts (2 weeks)
_SPEED_STEPS = {"1x": 1, "24x": 24, "168x": 168}
_SEV_COLOR   = {
    "Low":      "#22c55e",
    "Moderate": "#f59e0b",
    "High":     "#f97316",
    "Critical": "#ef4444",
}
_CHART_COLOR = "#22d3ee"    # cyan
_FAULT_COLOR = "#ef4444"    # red
_IMPUTED_CLR = "#f59e0b"    # amber

# ── caching ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _stations() -> pd.DataFrame:
    return load_aws_dataset()


@st.cache_data(show_spinner=False)
def _series(station_id: str) -> pd.DataFrame:
    """Generate + cache the full 3-year hourly series for one station."""
    df = _stations()
    row = df[df["station_id"] == station_id].iloc[0]
    return generate_station_series(row)


# ── minimal CSS ──────────────────────────────────────────────────────────────
def _css() -> None:
    st.markdown("""
<style>
/* KPI tile */
.kpi{background:#0a1224;border:1px solid #1e3a52;border-radius:10px;
     padding:12px 16px;text-align:center;}
.kpi-val{font-family:'Orbitron',sans-serif;font-size:1.45rem;
         font-weight:700;color:#22d3ee;line-height:1;}
.kpi-lbl{font-family:'JetBrains Mono',monospace;font-size:0.58rem;
         letter-spacing:.08em;text-transform:uppercase;color:#64748b;
         margin-top:4px;}
/* mini-metric row */
.mm{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;}
.mm-item{flex:1;min-width:100px;background:#0a1224;
         border:1px solid #1e3a52;border-radius:8px;padding:8px 10px;}
.mm-val{font-family:'JetBrains Mono',monospace;font-size:0.82rem;
        font-weight:600;color:#c7d6e6;}
.mm-lbl{font-family:'JetBrains Mono',monospace;font-size:0.55rem;
        text-transform:uppercase;color:#64748b;margin-top:2px;}
/* AI card */
.ai{background:#0a1224;border:1px solid #1e3a52;border-radius:10px;
    padding:14px;}
.ai-h{font-family:'JetBrains Mono',monospace;font-size:0.65rem;
      letter-spacing:.08em;text-transform:uppercase;color:#64748b;
      margin-bottom:8px;}
.ai-row{display:flex;justify-content:space-between;
        padding:4px 0;border-bottom:1px solid #0f1f35;}
.ai-k{font-family:'JetBrains Mono',monospace;font-size:0.62rem;
      text-transform:uppercase;color:#64748b;}
.ai-v{font-family:'JetBrains Mono',monospace;font-size:0.72rem;
      font-weight:600;color:#c7d6e6;}
.ai-root{margin-top:10px;padding:8px 10px;
         border-left:3px solid #22d3ee;
         font-family:'JetBrains Mono',monospace;font-size:0.7rem;
         color:#8aa2bd;line-height:1.55;}
.ai-action{margin-top:8px;padding:8px 10px;background:#0d2035;
           border-radius:6px;font-family:'JetBrains Mono',monospace;
           font-size:0.68rem;color:#8aa2bd;line-height:1.5;}
/* section label */
.sec{font-family:'JetBrains Mono',monospace;font-size:0.6rem;
     letter-spacing:.08em;text-transform:uppercase;color:#64748b;
     margin-top:12px;margin-bottom:6px;}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def _val(row: pd.Series, col: str, fmt: str = ".1f", suffix: str = "") -> str:
    try:
        return f"{float(row[col]):{fmt}}{suffix}"
    except Exception:
        return "—"


def _kpi_html(val: str, lbl: str, color: str = "#22d3ee") -> str:
    return (f'<div class="kpi">'
            f'<div class="kpi-val" style="color:{color};">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>')


# ── charts ────────────────────────────────────────────────────────────────────
def _make_charts(
    gt: pd.DataFrame,
    faulty: Optional[pd.DataFrame],
    current_idx: int,
) -> go.Figure:
    """4-panel Plotly figure with optional fault + imputed traces."""
    w_start = max(0, current_idx - _WINDOW // 2)
    w_end   = min(len(gt), w_start + _WINDOW)
    gt_w    = gt.iloc[w_start:w_end]
    ts      = gt_w["timestamp"]
    cur_ts  = gt["timestamp"].iloc[current_idx]

    channels = [
        ("temperature",     "Temperature (°C)"),
        ("humidity",        "Humidity (%)"),
        ("pressure",        "Pressure (mbar)"),
        ("battery_voltage", "Battery (V)"),
    ]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[c[1] for c in channels],
    )

    for i, (col, label) in enumerate(channels, 1):
        # Ground truth
        fig.add_trace(go.Scatter(
            x=ts, y=gt_w[col], mode="lines", name="Ground Truth",
            line=dict(color=_CHART_COLOR, width=1.4),
            legendgroup="gt", showlegend=(i == 1),
        ), row=i, col=1)

        # Faulty overlay
        if faulty is not None:
            f_w = faulty.iloc[w_start:w_end]
            fig.add_trace(go.Scatter(
                x=ts, y=f_w[col], mode="lines", name="Faulty",
                line=dict(color=_FAULT_COLOR, width=1.4),
                legendgroup="fault", showlegend=(i == 1),
            ), row=i, col=1)
            # AI imputed (linear fill over NaN, else blend toward GT)
            imp = f_w[col].copy().interpolate(method="linear").fillna(gt_w[col])
            fig.add_trace(go.Scatter(
                x=ts, y=imp, mode="lines", name="AI Imputed",
                line=dict(color=_IMPUTED_CLR, width=1.1, dash="dot"),
                legendgroup="imp", showlegend=(i == 1),
            ), row=i, col=1)

        # Vertical playback cursor
        y_min = float(gt_w[col].min())
        y_max = float(gt_w[col].max())
        fig.add_trace(go.Scatter(
            x=[cur_ts, cur_ts], y=[y_min, y_max],
            mode="lines", line=dict(color="#ef4444", width=1, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ), row=i, col=1)

    fig.update_layout(
        height=680,
        paper_bgcolor="#070d1a",
        plot_bgcolor="#0a1224",
        margin=dict(l=0, r=0, t=28, b=0),
        legend=dict(
            orientation="h", x=0, y=1.015,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b", size=10, family="JetBrains Mono, monospace"),
        ),
        font=dict(color="#64748b", family="JetBrains Mono, monospace", size=9),
    )
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor="#0f1f35", linecolor="#1e3a52",
            showticklabels=(i == 4), row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor="#0f1f35", linecolor="#1e3a52", row=i, col=1,
        )

    return fig


# ── AI detection card ─────────────────────────────────────────────────────────
def _ai_card(det: Optional[dict]) -> None:
    st.markdown('<div class="sec">🤖 AI Anomaly Detection</div>', unsafe_allow_html=True)
    if det is None:
        st.markdown(
            '<div class="ai"><div style="text-align:center;padding:24px 8px;'
            'font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#1e3a52;">'
            'No fault injected</div></div>',
            unsafe_allow_html=True,
        )
        return

    sev   = det.get("severity", "Unknown")
    color = _SEV_COLOR.get(sev, "#64748b")
    conf  = float(det.get("confidence", 0))

    rows = "".join(
        f'<div class="ai-row"><span class="ai-k">{k}</span>'
        f'<span class="ai-v" style="color:{color if k in ('Severity',) else '#c7d6e6'};">{v}</span></div>'
        for k, v in [
            ("Fault",    det.get("fault_type", "—")),
            ("Detected", det.get("detected_at", "—")),
            ("Latency",  f"{det.get('latency_minutes', 0)} min"),
            ("Severity", sev),
        ]
    )

    bg = f"rgba({','.join(str(int(color[i:i+2],16)) for i in (1,3,5))},0.12)"

    st.markdown(
        f'<div class="ai" style="border-color:{color}33;">'
        f'<div class="ai-h">Detection Result</div>'
        f'{rows}'
        f'<div style="margin-top:8px;background:{bg};border-radius:6px;'
        f'height:8px;overflow:hidden;">'
        f'<div style="width:{conf:.0f}%;height:100%;background:{color};'
        f'border-radius:6px;"></div></div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;'
        f'color:{color};text-align:right;margin-top:3px;">{conf:.1f}% confidence</div>'
        f'<div class="ai-root"><b>Root cause:</b><br>{det.get("root_cause","—")}</div>'
        f'<div class="ai-action">⚡ {det.get("action","—")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── main page ─────────────────────────────────────────────────────────────────
def render_simulation() -> None:
    _css()
    state    = get_state()
    stations = _stations()

    # ── page title ────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-family:Orbitron,sans-serif;font-size:1.2rem;"
        "font-weight:700;color:#e7f6ff;'>🔬 Simulation Lab</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;"
        "color:#64748b;text-transform:uppercase;letter-spacing:.07em;"
        "margin-bottom:10px;'>3-Year AWS Digital Twin · 826 Stations · "
        "Fault Injection · AI Detection</div>",
        unsafe_allow_html=True,
    )

    # ── TOP: Controls (horizontal) ────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns([3, 1.2, 1.2, 1, 0.8, 0.8, 0.8])

    with c1:
        all_ids    = stations["station_id"].tolist()
        all_labels = (
            stations["station_name"] + " · " + stations["state"]
        ).tolist()
        cur_sid = state.get("station_id") or all_ids[0]
        cur_idx = all_ids.index(cur_sid) if cur_sid in all_ids else 0
        sel_lbl = st.selectbox(
            "Station", all_labels, index=cur_idx,
            key="sl_station", label_visibility="collapsed",
        )
        sel_sid = all_ids[all_labels.index(sel_lbl)]
        if sel_sid != state.get("station_id"):
            state["station_id"]  = sel_sid
            state["current_idx"] = 0
            state["fault_active"]= False
            state["fault_result"]= None

    with c2:
        import datetime
        d_start = st.date_input(
            "From", value=datetime.date(2023, 1, 1),
            min_value=datetime.date(2023, 1, 1),
            max_value=datetime.date(2025, 12, 30),
            key="sl_from", label_visibility="collapsed",
        )
    with c3:
        d_end = st.date_input(
            "To", value=datetime.date(2023, 12, 31),
            min_value=datetime.date(2023, 1, 2),
            max_value=datetime.date(2025, 12, 31),
            key="sl_to", label_visibility="collapsed",
        )
    with c4:
        speed = st.select_slider(
            "Speed", ["1x", "24x", "168x"],
            value=state.get("speed", "24x"),
            key="sl_speed", label_visibility="collapsed",
        )
        state["speed"] = speed

    with c5:
        play_lbl = "⏸ Pause" if state["playing"] else "▶ Play"
        if st.button(play_lbl, key="sl_play", use_container_width=True):
            state["playing"] = not state["playing"]

    with c6:
        if st.button("↺ Reset", key="sl_reset", use_container_width=True):
            reset_state()
            st.rerun()

    with c7:
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
            f'color:#38bdf8;padding-top:8px;text-align:center;">'
            f'{"▶ LIVE" if state["playing"] else "⏸ PAUSED"}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#0f1f35;margin:8px 0;'>", unsafe_allow_html=True)

    # ── Load series ───────────────────────────────────────────────────────────
    with st.spinner("Loading telemetry…"):
        gt = _series(sel_sid)

    # Clip current_idx to date-range selection
    ts_arr    = pd.to_datetime(gt["timestamp"])
    idx_start = int((ts_arr >= pd.Timestamp(d_start)).idxmax())
    idx_end   = max(idx_start + 1,
                    int((ts_arr <= pd.Timestamp(d_end) + pd.Timedelta(hours=23)).sum()))
    idx_end   = min(idx_end, len(gt) - 1)

    cur_idx = int(state.get("current_idx", idx_start))
    cur_idx = max(idx_start, min(cur_idx, idx_end))
    state["current_idx"] = cur_idx

    cur_row = gt.iloc[cur_idx]
    cur_ts  = cur_row["timestamp"]

    # Determine active data (faulty or ground truth)
    faulty_df: Optional[pd.DataFrame] = None
    if state.get("fault_active"):
        fi = state.get("fault_inject_idx", cur_idx)
        fd = state.get("fault_df_json")
        if fd is not None:
            faulty_df = pd.read_json(fd, orient="split")
            faulty_df["timestamp"] = pd.to_datetime(faulty_df["timestamp"])
        cur_row = faulty_df.iloc[cur_idx] if faulty_df is not None else cur_row

    # ── Main layout: chart area | right panel ─────────────────────────────────
    main_col, right_col = st.columns([3, 1.15], gap="medium")

    with main_col:
        # ── 4 KPI cards ──────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(_kpi_html(_val(cur_row, "temperature", ".1f", " °C"), "Temperature"), unsafe_allow_html=True)
        with k2:
            st.markdown(_kpi_html(_val(cur_row, "humidity", ".0f", " %"), "Humidity", "#22c55e"), unsafe_allow_html=True)
        with k3:
            st.markdown(_kpi_html(_val(cur_row, "pressure", ".1f", " mbar"), "Pressure", "#a78bfa"), unsafe_allow_html=True)
        with k4:
            bat = float(cur_row.get("battery_voltage", 12.5))
            bat_c = "#22c55e" if bat > 12.3 else ("#f59e0b" if bat > 11.5 else "#ef4444")
            st.markdown(_kpi_html(f"{bat:.2f} V", "Battery", bat_c), unsafe_allow_html=True)

        # ── mini metrics ─────────────────────────────────────────────────────
        mm_html = (
            f'<div class="mm">'
            f'<div class="mm-item"><div class="mm-val">{_val(cur_row,"wind_speed",".1f")} m/s</div>'
            f'<div class="mm-lbl">Wind Speed</div></div>'
            f'<div class="mm-item"><div class="mm-val">{_val(cur_row,"rainfall",".1f")} mm</div>'
            f'<div class="mm-lbl">Rainfall</div></div>'
            f'<div class="mm-item"><div class="mm-val">{_val(cur_row,"signal_strength",".1f")} dBm</div>'
            f'<div class="mm-lbl">Signal</div></div>'
            f'<div class="mm-item"><div class="mm-val">{_val(cur_row,"solar_voltage",".2f")} V</div>'
            f'<div class="mm-lbl">Solar</div></div>'
            f'</div>'
        )
        st.markdown(mm_html, unsafe_allow_html=True)

        # Current timestamp display
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
            f'color:#38bdf8;margin-top:6px;margin-bottom:4px;">'
            f'📅 {cur_ts.strftime("%A, %d %b %Y  %H:%M")} IST'
            f'  &nbsp;·&nbsp; Row {cur_idx:,} / {idx_end:,}</div>',
            unsafe_allow_html=True,
        )

        # ── 4 synchronized charts ─────────────────────────────────────────────
        fig = _make_charts(gt, faulty_df, cur_idx)
        st.plotly_chart(fig, width="stretch", config=_CHART_CFG, key="sl_charts")

        # Station info strip
        sel_station = stations[stations["station_id"] == sel_sid].iloc[0]
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;'
            f'color:#1e3a52;margin-top:4px;">'
            f'{sel_sid} · {sel_station.get("state","")}'
            f' · Lat {sel_station.get("latitude",0):.4f}°N'
            f' · Lon {sel_station.get("longitude",0):.4f}°E'
            f' · Elev {sel_station.get("elevation",0):.0f} m</div>',
            unsafe_allow_html=True,
        )

    # ── RIGHT: Fault injection + AI inspector ─────────────────────────────────
    with right_col:

        # Station snapshot
        hs = float(stations[stations["station_id"] == sel_sid]["health_score"].iloc[0])
        hs_c = "#22c55e" if hs >= 75 else ("#f59e0b" if hs >= 55 else "#ef4444")
        st.markdown(
            f'<div style="background:#0a1224;border:1px solid #1e3a52;'
            f'border-radius:10px;padding:12px;margin-bottom:8px;">'
            f'<div style="font-family:Orbitron,sans-serif;font-size:0.82rem;'
            f'font-weight:700;color:#e7f6ff;margin-bottom:2px;">'
            f'{sel_station.get("station_name","")}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;'
            f'color:#64748b;">{sel_station.get("state","")} · '
            f'{sel_station.get("district","")}</div>'
            f'<div style="margin-top:8px;background:#0f1f35;height:6px;border-radius:3px;">'
            f'<div style="width:{hs:.0f}%;height:100%;background:{hs_c};border-radius:3px;"></div></div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;'
            f'color:{hs_c};text-align:right;margin-top:2px;">Health {hs:.1f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Fault injection panel
        st.markdown('<div class="sec">⚡ Fault Injection</div>', unsafe_allow_html=True)

        fault_type = st.selectbox(
            "Fault Type", FAULT_TYPES,
            index=FAULT_TYPES.index(state.get("fault_type", FAULT_TYPES[0])),
            key="sl_ftype", label_visibility="collapsed",
        )
        state["fault_type"] = fault_type

        intensity = st.slider(
            "Intensity", 1, 10,
            value=int(state.get("fault_intensity", 5)),
            key="sl_intensity",
        )
        state["fault_intensity"] = intensity

        duration = st.number_input(
            "Duration (hours)", min_value=1, max_value=720,
            value=int(state.get("fault_duration", 48)),
            step=6, key="sl_duration",
        )
        state["fault_duration"] = int(duration)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💥 Inject", key="sl_inject", use_container_width=True):
                f_df, det = inject_fault(
                    gt,
                    fault_type=fault_type,
                    intensity=intensity,
                    duration_hours=int(duration),
                    start_idx=cur_idx,
                )
                state["fault_active"]     = True
                state["fault_result"]     = det
                state["fault_inject_idx"] = cur_idx
                state["fault_df_json"]    = f_df.to_json(orient="split",
                                                          date_format="iso")
                st.rerun()

        with b2:
            if st.button("✖ Clear", key="sl_clear", use_container_width=True):
                state["fault_active"]  = False
                state["fault_result"]  = None
                state["fault_df_json"] = None
                st.rerun()

        # AI detection card
        _ai_card(state.get("fault_result"))

        # Legend
        st.markdown('<div class="sec" style="margin-top:14px;">Chart Legend</div>',
                    unsafe_allow_html=True)
        for lbl, col in [("Ground Truth", _CHART_COLOR),
                         ("Faulty",       _FAULT_COLOR),
                         ("AI Imputed",   _IMPUTED_CLR),
                         ("Playback",     "#ef4444")]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:7px;margin:3px 0;">'
                f'<div style="width:20px;height:2px;background:{col};flex-shrink:0;"></div>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:0.62rem;'
                f'color:#64748b;">{lbl}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Playback advancement ──────────────────────────────────────────────────
    if state["playing"]:
        step = _SPEED_STEPS.get(speed, 24)
        new_idx = cur_idx + step
        if new_idx >= idx_end:
            state["playing"]     = False
            state["current_idx"] = idx_end
        else:
            state["current_idx"] = new_idx
            time.sleep(0.08)
            st.rerun()
