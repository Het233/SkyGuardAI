"""
================================================================================
dashboard/pages/network_map.py -- SkyGuard AI  |  Network Map  (826 AWS)
================================================================================
Renders the India AWS Network page backed by the real 826-station EDA dataset
loaded from data_loader.load_aws_dataset().

Layout
------
  KPI row  (Total AWS | Visible | Critical | Avg Health)
  Filter toolbar  (State | Status | Search | Health slider)  -- horizontal
  Map (60%) | Station Inspector (40%)
  Bottom analytics strip  (Elevation hist | Stations/State | Temp vs Elev)

Entry-point (backward-compatible with app.py signature):
    render_network_map(demo_data=None, station_metadata=None)
================================================================================
"""

from __future__ import annotations

import datetime
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Resolve data_loader from the dashboard package root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import load_aws_dataset

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
STATUS_COLORS: Dict[str, str] = {
    "EXCELLENT": "#22c55e",
    "GOOD":      "#38bdf8",
    "DEGRADING": "#f59e0b",
    "CRITICAL":  "#ef4444",
}
_MAP_STYLE = "carto-darkmatter"


# ---------------------------------------------------------------------------
# Cached data fetch
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_station_data() -> pd.DataFrame:
    """Return the full 826-station dataframe, cached for the session lifetime."""
    return load_aws_dataset()


# ---------------------------------------------------------------------------
# CSS  (minimal -- respects existing SkyGuard theme)
# ---------------------------------------------------------------------------
def _css() -> None:
    st.markdown(
        """
<style>
/* ---- KPI cards ---- */
.nw-kpi-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px;}
.nw-kpi{
  flex:1;min-width:130px;
  background:rgba(14,22,40,0.65);
  border:1px solid rgba(56,227,255,0.13);
  border-radius:14px;padding:14px 16px;
}
.nw-kpi-val{
  font-family:'Orbitron',sans-serif;font-size:1.55rem;font-weight:700;
  color:#38bdf8;line-height:1;
}
.nw-kpi-label{
  font-family:'JetBrains Mono',monospace;font-size:0.62rem;
  letter-spacing:.07em;text-transform:uppercase;color:#8aa2bd;margin-top:4px;
}

/* ---- Glass panel ---- */
.nw-panel{
  background:rgba(14,22,40,0.55);
  border:1px solid rgba(255,255,255,0.07);
  border-radius:16px;padding:16px;
}

/* ---- Inspector card ---- */
.si-card{
  background:rgba(8,13,24,0.7);
  border:1px solid rgba(56,227,255,0.14);
  border-radius:14px;padding:18px;
}
.si-title{
  font-family:'Orbitron',sans-serif;font-size:0.9rem;
  font-weight:700;color:#e7f6ff;margin-bottom:2px;
}
.si-sub{
  font-family:'Rajdhani',sans-serif;font-size:0.82rem;
  color:#8aa2bd;margin-bottom:14px;
}
.si-row{
  display:flex;justify-content:space-between;
  border-bottom:1px solid rgba(255,255,255,0.05);
  padding:5px 0;
}
.si-key{
  font-family:'JetBrains Mono',monospace;font-size:0.64rem;
  letter-spacing:.06em;text-transform:uppercase;color:#54637a;
}
.si-val{
  font-family:'JetBrains Mono',monospace;font-size:0.76rem;
  font-weight:600;color:#c7d6e6;
}
.si-badge{
  display:inline-block;padding:3px 10px;border-radius:999px;
  font-family:'JetBrains Mono',monospace;font-size:0.65rem;
  font-weight:700;letter-spacing:.06em;margin-top:12px;
}
.si-empty{
  text-align:center;padding:40px 10px;
  font-family:'JetBrains Mono',monospace;font-size:0.76rem;color:#54637a;
}

/* ---- Section label ---- */
.nw-sec{
  font-family:'JetBrains Mono',monospace;font-size:0.65rem;
  letter-spacing:.09em;text-transform:uppercase;color:#8aa2bd;
  margin-bottom:8px;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(v: Any, decimals: int = 1, suffix: str = "") -> str:
    try:
        return f"{float(v):.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def _fmt_int(v: Any) -> str:
    try:
        return f"{int(v):,}"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
def _kpi_row(df_all: pd.DataFrame, df_vis: pd.DataFrame) -> None:
    total   = len(df_all)
    visible = len(df_vis)
    critical = int((df_vis["status"] == "CRITICAL").sum())
    avg_h   = df_vis["health_score"].mean() if not df_vis.empty else 0.0

    kpis = [
        (_fmt_int(total),   "Total AWS Stations"),
        (_fmt_int(visible), "Visible Stations"),
        (_fmt_int(critical),"Critical"),
        (_fmt(avg_h, 1),    "Avg Health Score"),
    ]
    html = '<div class="nw-kpi-row">' + "".join(
        f'<div class="nw-kpi">'
        f'<div class="nw-kpi-val">{v}</div>'
        f'<div class="nw-kpi-label">{l}</div>'
        f'</div>'
        for v, l in kpis
    ) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Filter toolbar  (single horizontal row)
# ---------------------------------------------------------------------------
def _filter_toolbar(df: pd.DataFrame):
    """Render horizontal filter row; return filtered dataframe."""
    fa, fb, fc, fd = st.columns([2, 2, 2, 3])

    all_states  = sorted(df["state"].dropna().unique().tolist())
    all_statuses= ["EXCELLENT", "GOOD", "DEGRADING", "CRITICAL"]

    with fa:
        sel_states = st.multiselect(
            "State / UT", options=all_states,
            default=[], key="nw_states",
            placeholder="All states",
            label_visibility="collapsed",
        )
    with fb:
        sel_status = st.multiselect(
            "Status", options=all_statuses,
            default=[], key="nw_status",
            placeholder="All statuses",
            label_visibility="collapsed",
        )
    with fc:
        search_id = st.text_input(
            "Search Station ID",
            value="", key="nw_search",
            placeholder="Station ID…",
            label_visibility="collapsed",
        )
    with fd:
        h_min = float(df["health_score"].min())
        h_max = float(df["health_score"].max())
        h_range = st.slider(
            "Health score range", min_value=h_min, max_value=h_max,
            value=(h_min, h_max), key="nw_health",
            label_visibility="collapsed",
        )

    # Apply filters in-memory
    mask = pd.Series(True, index=df.index)
    if sel_states:
        mask &= df["state"].isin(sel_states)
    if sel_status:
        mask &= df["status"].isin(sel_status)
    if search_id.strip():
        q = search_id.strip().lower()
        mask &= (
            df["station_id"].str.lower().str.contains(q, na=False)
            | df["station_name"].str.lower().str.contains(q, na=False)
        )
    mask &= (df["health_score"] >= h_range[0]) & (df["health_score"] <= h_range[1])

    return df[mask].reset_index(drop=True), search_id.strip()


# ---------------------------------------------------------------------------
# Scattermap  (modern Plotly >= 5.24 API -- replaces deprecated Scattermapbox)
# ---------------------------------------------------------------------------
def _build_map(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            map=dict(style=_MAP_STYLE, center=dict(lat=20.5, lon=78.9), zoom=4),
            height=540, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    marker_colors = df["status"].map(STATUS_COLORS).fillna("#8aa2bd").tolist()
    marker_sizes  = (6 + (100 - df["health_score"].clip(0, 100)) * 0.08).tolist()

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "State: %{customdata[2]}<br>"
        "Temperature: %{customdata[3]} \u00b0C<br>"
        "Humidity: %{customdata[4]} %<br>"
        "Pressure: %{customdata[5]} mbar<br>"
        "Health: %{customdata[6]}<br>"
        "Status: %{customdata[7]}"
        "<extra></extra>"
    )

    custom = list(zip(
        df["station_id"],
        df["station_name"],
        df["state"],
        df["temperature"].round(1).astype(str),
        df["humidity"].round(1).astype(str),
        df["pressure"].round(1).astype(str),
        df["health_score"].round(1).astype(str),
        df["status"],
    ))

    fig = go.Figure(go.Scattermap(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        marker=dict(
            color=marker_colors,
            size=marker_sizes,
            opacity=0.85,
        ),
        customdata=custom,
        hovertemplate=hover,
        hoverlabel=dict(
            bgcolor="rgba(6,10,20,0.96)",
            bordercolor="rgba(56,189,248,0.45)",
            font=dict(color="#e7f6ff", family="JetBrains Mono, monospace", size=11),
        ),
    ))

    fig.update_layout(
        map=dict(
            style=_MAP_STYLE,
            center=dict(lat=20.5, lon=78.9),
            zoom=4.0,
        ),
        height=540,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision="network_map_stable",
    )
    return fig


# ---------------------------------------------------------------------------
# Station Inspector
# ---------------------------------------------------------------------------
def _inspector(row: Optional[pd.Series]) -> None:
    st.markdown('<div class="nw-sec">🔎 Station Inspector</div>', unsafe_allow_html=True)

    if row is None:
        st.markdown(
            '<div class="si-card"><div class="si-empty">'
            'No station selected.<br>Apply a search filter<br>to inspect a station.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    status = str(row.get("status", "GOOD"))
    color  = STATUS_COLORS.get(status, "#38bdf8")
    hs     = float(row.get("health_score", 75))
    bg     = "rgba({},{},{},0.14)".format(*[int(color[i:i+2], 16) for i in (1, 3, 5)])

    rows_html = "".join(
        f'<div class="si-row"><span class="si-key">{k}</span>'
        f'<span class="si-val">{v}</span></div>'
        for k, v in [
            ("Station ID",   row.get("station_id",   "N/A")),
            ("Station Name", row.get("station_name", "N/A")),
            ("State",        row.get("state",        "N/A")),
            ("District",     row.get("district",     "N/A")),
            ("Temperature",  f"{_fmt(row.get('temperature'))} °C"),
            ("Humidity",     f"{_fmt(row.get('humidity'))} %"),
            ("Pressure",     f"{_fmt(row.get('pressure'))} mbar"),
            ("Elevation",    f"{_fmt(row.get('elevation'), 0)} m"),
        ]
    )

    badge = (
        f'<span class="si-badge" '
        f'style="background:{bg};border:1px solid {color};color:{color};">'
        f'{status}</span>'
    )

    pct = min(100, max(0, hs))
    bar = (
        f'<div style="margin-top:14px;">'
        f'<div class="si-key" style="margin-bottom:4px;">Health Score</div>'
        f'<div style="background:rgba(255,255,255,0.06);border-radius:6px;height:8px;overflow:hidden;">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color};'
        f'border-radius:6px;transition:width 0.4s ease;"></div></div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
        f'color:{color};margin-top:4px;text-align:right;">{pct:.1f}</div>'
        f'</div>'
    )

    st.markdown(
        f'<div class="si-card">'
        f'<div class="si-title">{row.get("station_name", "N/A")}</div>'
        f'<div class="si-sub">{row.get("state", "")} · {row.get("district", "")}</div>'
        f'{rows_html}{bar}{badge}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Bottom analytics
# ---------------------------------------------------------------------------
_CHART_CFG = dict(displayModeBar=False)


def _bottom_analytics(df: pd.DataFrame) -> None:
    if df.empty:
        return
    c1, c2, c3 = st.columns(3)

    # 1. Elevation histogram
    with c1:
        st.markdown('<div class="nw-sec">📊 Elevation Distribution</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(
            df, x="elevation", nbins=40,
            color_discrete_sequence=["#38bdf8"],
            labels={"elevation": "Elevation (m)"},
        )
        fig.update_layout(
            height=240,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=4,b=0),
            bargap=0.06,
            xaxis=dict(color="#8aa2bd", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(color="#8aa2bd", gridcolor="rgba(255,255,255,0.05)"),
            font=dict(family="JetBrains Mono, monospace", color="#8aa2bd", size=10),
        )
        st.plotly_chart(fig, width="stretch", config=_CHART_CFG, key="nw_elev_hist")

    # 2. Stations per state (top 15)
    with c2:
        st.markdown('<div class="nw-sec">🗺️ Stations per State (Top 15)</div>',
                    unsafe_allow_html=True)
        top = (
            df["state"].value_counts().head(15).reset_index()
            .rename(columns={"state": "State", "count": "Stations"})
        )
        fig2 = px.bar(
            top, x="Stations", y="State", orientation="h",
            color="Stations", color_continuous_scale="Blues",
        )
        fig2.update_layout(
            height=240,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=4,b=0),
            yaxis=dict(autorange="reversed", color="#8aa2bd",
                       gridcolor="rgba(255,255,255,0.05)",
                       tickfont=dict(size=9)),
            xaxis=dict(color="#8aa2bd", gridcolor="rgba(255,255,255,0.05)"),
            coloraxis_showscale=False,
            font=dict(family="JetBrains Mono, monospace", color="#8aa2bd", size=10),
        )
        st.plotly_chart(fig2, width="stretch", config=_CHART_CFG, key="nw_state_bar")

    # 3. Temperature vs Elevation
    with c3:
        st.markdown('<div class="nw-sec">🌡️ Temperature vs Elevation</div>',
                    unsafe_allow_html=True)
        sample = df.sample(min(600, len(df)), random_state=42) if len(df) > 600 else df
        fig3 = px.scatter(
            sample, x="elevation", y="temperature",
            color="status",
            color_discrete_map=STATUS_COLORS,
            labels={"elevation": "Elevation (m)", "temperature": "Temp (°C)"},
            opacity=0.72,
        )
        fig3.update_layout(
            height=240,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=4,b=0),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=8, color="#8aa2bd"),
                title_text="",
            ),
            xaxis=dict(color="#8aa2bd", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(color="#8aa2bd", gridcolor="rgba(255,255,255,0.05)"),
            font=dict(family="JetBrains Mono, monospace", color="#8aa2bd", size=10),
        )
        fig3.update_traces(marker=dict(size=5))
        st.plotly_chart(fig3, width="stretch", config=_CHART_CFG, key="nw_temp_scatter")


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
def render_network_map(
    demo_data: Any = None,
    station_metadata: Any = None,
) -> None:
    """Render the SkyGuard AI 826-AWS Network Map page.

    ``demo_data`` and ``station_metadata`` are accepted for backward
    compatibility with app.py but are ignored -- data comes from the EDA
    dataset via get_station_data().
    """
    _css()

    # --- Load data ----------------------------------------------------------
    df_all = get_station_data()

    # --- Page heading -------------------------------------------------------
    st.markdown(
        """
<div style="font-family:'Orbitron',sans-serif;font-size:1.25rem;font-weight:700;
     color:#e7f6ff;letter-spacing:0.03em;display:flex;align-items:center;gap:10px;">
  📡&nbsp; Network Map — India AWS Fleet
</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
     color:#8aa2bd;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:14px;">
  Automatic Weather Stations · IMD · 826 stations · All 36 States &amp; UTs
</div>
""",
        unsafe_allow_html=True,
    )

    # --- Filter toolbar (horizontal) ----------------------------------------
    df_vis, search_q = _filter_toolbar(df_all)

    # --- KPI row ------------------------------------------------------------
    _kpi_row(df_all, df_vis)

    st.markdown("---")

    # --- Map | Inspector layout ---------------------------------------------
    map_col, insp_col = st.columns([3, 1.6], gap="medium")

    with map_col:
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
            f'color:#8aa2bd;text-align:right;margin-bottom:4px;">'
            f'{len(df_vis):,}&nbsp;/&nbsp;{len(df_all):,}&nbsp;STATIONS SHOWN</div>',
            unsafe_allow_html=True,
        )
        fig_map = _build_map(df_vis)
        st.plotly_chart(
            fig_map,
            width="stretch",
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["toImage"],
                "displaylogo": False,
                "scrollZoom": True,
            },
            key="nw_mapbox_main",
        )

        # Status legend chips
        legend = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'font-family:JetBrains Mono,monospace;font-size:0.63rem;color:{c};'
            f'background:rgba({",".join(str(int(c[i:i+2],16)) for i in (1,3,5))},0.12);'
            f'border:1px solid {c};border-radius:999px;padding:3px 9px;margin-right:6px;">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:{c};"></span>'
            f'{s}&nbsp;({int((df_vis["status"]==s).sum())})</span>'
            for s, c in STATUS_COLORS.items()
            if (df_vis["status"] == s).any()
        )
        if legend:
            st.markdown(
                f'<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;">{legend}</div>',
                unsafe_allow_html=True,
            )

    with insp_col:
        # Determine which station to show in inspector
        insp_row: Optional[pd.Series] = None
        if not df_vis.empty:
            if search_q:
                q = search_q.lower()
                hit = df_vis[
                    df_vis["station_id"].str.lower().str.contains(q, na=False)
                    | df_vis["station_name"].str.lower().str.contains(q, na=False)
                ]
                insp_row = hit.iloc[0] if not hit.empty else df_vis.iloc[0]
            else:
                insp_row = df_vis.iloc[0]

        _inspector(insp_row)

    # --- Bottom analytics ---------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="nw-sec">📈 Network Analytics</div>', unsafe_allow_html=True)
    _bottom_analytics(df_vis)
