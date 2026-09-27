"""
================================================================================
dashboard/pages/big_data_analytics.py  —  SkyGuard AI
Big Data Climate Intelligence Platform
================================================================================
Executive analytics dashboard covering all 826 IMD AWS stations.
Sections:
  1. Big Data KPI strip
  2. Geospatial choropleth by state (metric-switchable)
  3. Hive Query Explorer (SQL-style analytics on pandas)
  4. Spark Analytics (4 interactive charts, state-filtered)
  5. Data Quality Engine (donut + heatmap + gauge)
  6. Stream Processing monitor (session_state, no loops)
  7. Climate Insights (auto-generated from dataset)

Entry-point:
    render_big_data_analytics()
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
import hashlib
import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── path resolution ───────────────────────────────────────────────────────
_DASH = os.path.dirname(os.path.dirname(__file__))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

from data_loader import load_aws_dataset
from simulation.weather_generator import generate_station_series

# ── shared chart config ──────────────────────────────────────────────────
_CFG  = {"displayModeBar": False, "responsive": True}
_CYAN = "#22d4ee"
_DIM  = "#8aa2bd"
_BG   = "rgba(0,0,0,0)"
_GRID = "rgba(255,255,255,0.06)"
_PLOT_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(family="JetBrains Mono, monospace", color=_DIM, size=11),
    margin=dict(l=12, r=12, t=36, b=12),
)

# Months for rainfall chart
_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"]


# ───────────────────────────────────────────────────────────────────────────
# CSS
# ───────────────────────────────────────────────────────────────────────────

def _css() -> None:
    st.markdown("""
<style>
/* ---- Big Data Analytics page ---- */
.bd-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #22d4ee;
    margin: 1.6rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    border-left: 3px solid #22d4ee;
    padding-left: 0.7rem;
}
.bd-section-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.78rem;
    color: #54637a;
    margin-left: auto;
    text-transform: none;
    letter-spacing: 0;
}

/* KPI cards */
.bd-kpi {
    background: rgba(8,14,28,0.85);
    border: 1px solid rgba(34,212,238,0.15);
    border-radius: 12px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.bd-kpi:hover { border-color: rgba(34,212,238,0.4); }
.bd-kpi::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--kc, #22d4ee);
    opacity: 0.6;
}
.bd-kpi-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--kc, #22d4ee);
    line-height: 1.1;
    word-break: break-all;
}
.bd-kpi-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #54637a;
    margin-top: 5px;
}
.bd-kpi-icon {
    font-size: 1.1rem;
    margin-bottom: 6px;
    display: block;
}

/* SQL block */
.bd-sql {
    background: rgba(4,10,22,0.9);
    border: 1px solid rgba(34,212,238,0.15);
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #c7d6e6;
    line-height: 1.8;
    white-space: pre;
    overflow-x: auto;
    margin-bottom: 12px;
}
.bd-sql .kw { color: #22d4ee; }
.bd-sql .fn { color: #b084fc; }
.bd-sql .tb { color: #2bffa8; }
.bd-sql .cm { color: #54637a; }

/* Insight card */
.bd-insight {
    background: rgba(8,14,28,0.7);
    border: 1px solid rgba(34,212,238,0.12);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.92rem;
    color: #c7d6e6;
    line-height: 1.5;
}
.bd-insight-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.bd-insight-cat {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #54637a;
    margin-bottom: 2px;
}

/* Stream cards */
.bd-stream {
    background: rgba(8,14,28,0.85);
    border: 1px solid rgba(34,212,238,0.15);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.bd-stream-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #22d4ee;
    line-height: 1;
}
.bd-stream-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #54637a;
    margin-top: 5px;
}

/* Quality badge */
.bd-qbadge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 9px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.qb-green { background: rgba(43,255,168,0.10); border:1px solid rgba(43,255,168,0.4); color:#2bffa8; }
.qb-amber { background: rgba(255,200,87,0.10);  border:1px solid rgba(255,200,87,0.4);  color:#ffc857; }
.qb-red   { background: rgba(255,84,112,0.10);  border:1px solid rgba(255,84,112,0.4);  color:#ff5470; }
</style>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ───────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _stations() -> pd.DataFrame:
    """Load the 826-station EDA dataset, ensuring a deterministic wind/rainfall column."""
    df = load_aws_dataset().copy()
    # Derive deterministic wind_speed and rainfall per station from station_id seed
    seeds = [
        int(hashlib.md5(sid.encode()).hexdigest(), 16) % (2**31)
        for sid in df["station_id"]
    ]
    rngs = [np.random.default_rng(s) for s in seeds]
    df["wind_speed"] = [round(float(r.uniform(1.0, 12.0)), 1) for r in rngs]
    # Re-seed for rainfall
    rngs2 = [np.random.default_rng(s + 1) for s in seeds]
    df["rainfall"]   = [round(float(r.uniform(0.0, 25.0)), 1) for r in rngs2]
    return df


@st.cache_data(show_spinner=False)
def _sample_timeseries() -> pd.DataFrame:
    """
    Generate a compact multi-station monthly time series for trend charts.
    Sample 12 stations (one per major region) and produce hourly data,
    then resample to monthly for performance.
    """
    df_st = _stations()
    # Pick one station per state (up to 12 for speed)
    sampled = (
        df_st.dropna(subset=["latitude", "longitude"])
        .groupby("state", sort=False)
        .first()
        .reset_index()
        .sample(min(12, df_st["state"].nunique()), random_state=42)
    )
    frames = []
    for _, row in sampled.iterrows():
        try:
            ts = generate_station_series(row)
            ts["state"] = row["state"]
            ts["station_name"] = row["station_name"]
            frames.append(ts)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["month"] = combined["timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        combined.groupby(["state", "month"])
        .agg(
            temperature=("temperature", "mean"),
            humidity=("humidity", "mean"),
            pressure=("pressure", "mean"),
            wind_speed=("wind_speed", "mean"),
            rainfall=("rainfall", "sum"),
        )
        .reset_index()
    )
    return monthly


# ───────────────────────────────────────────────────────────────────────────
# SECTION 1  —  BIG DATA KPIs
# ───────────────────────────────────────────────────────────────────────────

def _render_kpis(df: pd.DataFrame) -> None:
    n_stations = len(df)
    # 826 stations × 3 years × 365.25 days × 24 hours × 5 sensors
    total_records = n_stations * 3 * 365 * 24 * 5
    n_states  = df["state"].nunique()
    avg_temp  = df["temperature"].mean()
    active_streams = n_stations * 5          # 5 sensor channels per station
    quality_score  = df["health_score"].mean() if "health_score" in df.columns else 87.4

    kpis = [
        ("\U0001f4e1", f"{n_stations:,}",               "Total AWS Stations",    "#22d4ee"),
        ("\U0001f5c4", f"{total_records:,}",             "Total Records Processed","#818cf8"),
        ("\U0001f5fa", f"{n_states}",                   "States / UTs Covered",  "#2bffa8"),
        ("\U0001f321", f"{avg_temp:.1f} \u00b0C",        "Avg Temperature",       "#f59e0b"),
        ("\u26a1",     f"{active_streams:,}",            "Active Sensor Streams", "#b084fc"),
        ("\u2705",     f"{quality_score:.1f}%",          "Data Quality Score",    "#2bffa8"),
    ]

    cols = st.columns(6, gap="small")
    for col, (icon, val, lbl, color) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
<div class="bd-kpi" style="--kc:{color};">
  <span class="bd-kpi-icon">{icon}</span>
  <div class="bd-kpi-val">{val}</div>
  <div class="bd-kpi-lbl">{lbl}</div>
</div>""",
                unsafe_allow_html=True,
            )


# ───────────────────────────────────────────────────────────────────────────
# SECTION 2  —  GEOSPATIAL ANALYTICS
# ───────────────────────────────────────────────────────────────────────────

_GEO_METRICS = {
    "Average Temperature (\u00b0C)": "temperature",
    "Average Humidity (%)": "humidity",
    "Average Pressure (hPa)": "pressure",
    "Average Wind Speed (m/s)": "wind_speed",
    "Average Rainfall (mm)": "rainfall",
}
_GEO_SCALES = {
    "temperature": "RdYlBu_r",
    "humidity":    "Blues",
    "pressure":    "Viridis",
    "wind_speed":  "Teal",
    "rainfall":    "Blues",
}


def _render_geospatial(df: pd.DataFrame) -> None:
    metric_label = st.selectbox(
        "Select metric",
        list(_GEO_METRICS.keys()),
        key="bd_geo_metric",
        label_visibility="collapsed",
    )
    col = _GEO_METRICS[metric_label]
    colorscale = _GEO_SCALES.get(col, "Viridis")

    # Aggregate by state
    agg = (
        df.groupby("state")
        .agg(
            avg_val=(col, "mean"),
            stations=("station_id", "count"),
            highest=(col, "max"),
            lowest=(col, "min"),
            highest_station=("station_name", lambda x: x.iloc[df.loc[x.index, col].argmax()]
                             if col in df.columns else "N/A"),
            lowest_station=("station_name",  lambda x: x.iloc[df.loc[x.index, col].argmin()]
                             if col in df.columns else "N/A"),
        )
        .reset_index()
    )
    agg["avg_val"] = agg["avg_val"].round(2)

    unit_map = {
        "temperature": "\u00b0C", "humidity": "%", "pressure": " hPa",
        "wind_speed": " m/s", "rainfall": " mm",
    }
    unit = unit_map.get(col, "")

    agg["hover"] = (
        "<b>" + agg["state"] + "</b><br>"
        + "Stations: " + agg["stations"].astype(str) + "<br>"
        + "Avg: " + agg["avg_val"].astype(str) + unit + "<br>"
        + "Highest: " + agg["highest"].round(1).astype(str) + unit + "<br>"
        + "Lowest: "  + agg["lowest"].round(1).astype(str) + unit
    )

    # Scatter-geo (no GeoJSON needed — lat/lon of state centroids)
    state_geo = df.groupby("state")[["latitude", "longitude"]].mean().reset_index()
    agg = agg.merge(state_geo, on="state", how="left")

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lat=agg["latitude"],
        lon=agg["longitude"],
        mode="markers",
        marker=dict(
            size=agg["stations"].clip(3, 30) * 1.8,
            color=agg["avg_val"],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text=metric_label, font=dict(size=10, color=_DIM)),
                tickfont=dict(color=_DIM, size=9),
                len=0.65,
            ),
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            opacity=0.88,
        ),
        text=agg["hover"],
        hoverinfo="text",
        name="",
    ))

    fig.update_geos(
        scope="asia",
        center=dict(lat=22, lon=80),
        projection_scale=4.8,
        showland=True,
        landcolor="rgba(12,20,40,0.9)",
        showocean=True,
        oceancolor="rgba(6,10,22,0.95)",
        showlakes=False,
        showcountries=True,
        countrycolor="rgba(34,212,238,0.25)",
        showsubunits=True,
        subunitcolor="rgba(34,212,238,0.15)",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        **_PLOT_LAYOUT,
        height=480,
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(
            text=f"India AWS Network — {metric_label}",
            font=dict(size=12, color=_DIM),
            x=0.01,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config=_CFG)


# ───────────────────────────────────────────────────────────────────────────
# SECTION 3  —  HIVE QUERY EXPLORER
# ───────────────────────────────────────────────────────────────────────────

_HIVE_QUERIES = {
    "Average Temperature by State": {
        "sql": (
            "SELECT   state,\n"
            "         COUNT(station_id)          AS stations,\n"
            "         ROUND(AVG(temperature), 2) AS avg_temp_c\n"
            "FROM     aws_data\n"
            "GROUP BY state\n"
            "ORDER BY avg_temp_c DESC;"
        ),
        "exec": lambda df: (
            df.groupby("state")
            .agg(stations=("station_id", "count"), avg_temp_c=("temperature", "mean"))
            .round(2)
            .reset_index()
            .sort_values("avg_temp_c", ascending=False)
        ),
        "chart": "bar_h",
        "x": "avg_temp_c", "y": "state", "color": "#f59e0b",
        "xlabel": "Avg Temperature (\u00b0C)",
    },
    "Top 10 Hottest Stations": {
        "sql": (
            "SELECT   station_name, state, district,\n"
            "         ROUND(temperature, 2) AS temperature_c,\n"
            "         elevation\n"
            "FROM     aws_data\n"
            "ORDER BY temperature_c DESC\n"
            "LIMIT    10;"
        ),
        "exec": lambda df: (
            df.nlargest(10, "temperature")[
                ["station_name", "state", "district", "temperature", "elevation"]
            ].rename(columns={"temperature": "temperature_c"})
        ),
        "chart": "bar_v",
        "x": "station_name", "y": "temperature_c", "color": "#ff5470",
        "xlabel": "Temperature (\u00b0C)",
    },
    "Highest Elevation Stations": {
        "sql": (
            "SELECT   station_name, state,\n"
            "         elevation,\n"
            "         ROUND(temperature, 2) AS temperature_c,\n"
            "         ROUND(pressure, 2)    AS pressure_hpa\n"
            "FROM     aws_data\n"
            "ORDER BY elevation DESC\n"
            "LIMIT    15;"
        ),
        "exec": lambda df: (
            df.nlargest(15, "elevation")[
                ["station_name", "state", "elevation", "temperature", "pressure"]
            ].rename(columns={"temperature": "temperature_c", "pressure": "pressure_hpa"})
        ),
        "chart": "scatter",
        "x": "elevation", "y": "temperature_c", "color": "#2bffa8",
        "xlabel": "Elevation (m)",
    },
    "Humidity Distribution by State": {
        "sql": (
            "SELECT   state,\n"
            "         ROUND(AVG(humidity), 2)  AS avg_humidity,\n"
            "         ROUND(MIN(humidity), 2)  AS min_humidity,\n"
            "         ROUND(MAX(humidity), 2)  AS max_humidity\n"
            "FROM     aws_data\n"
            "GROUP BY state\n"
            "ORDER BY avg_humidity DESC;"
        ),
        "exec": lambda df: (
            df.groupby("state")
            .agg(
                avg_humidity=("humidity", "mean"),
                min_humidity=("humidity", "min"),
                max_humidity=("humidity", "max"),
            )
            .round(2)
            .reset_index()
            .sort_values("avg_humidity", ascending=False)
        ),
        "chart": "bar_h",
        "x": "avg_humidity", "y": "state", "color": "#38bdf8",
        "xlabel": "Avg Humidity (%)",
    },
    "Monthly Rainfall Analysis": {
        "sql": (
            "SELECT   state,\n"
            "         ROUND(AVG(rainfall), 2)  AS avg_rainfall_mm,\n"
            "         COUNT(station_id)         AS station_count\n"
            "FROM     aws_data\n"
            "GROUP BY state\n"
            "ORDER BY avg_rainfall_mm DESC;"
        ),
        "exec": lambda df: (
            df.groupby("state")
            .agg(avg_rainfall_mm=("rainfall", "mean"), station_count=("station_id", "count"))
            .round(2)
            .reset_index()
            .sort_values("avg_rainfall_mm", ascending=False)
        ),
        "chart": "bar_h",
        "x": "avg_rainfall_mm", "y": "state", "color": "#818cf8",
        "xlabel": "Avg Rainfall (mm)",
    },
    "Heatwave Frequency (T > 40\u00b0C)": {
        "sql": (
            "SELECT   state,\n"
            "         COUNT(station_id)           AS heatwave_stations,\n"
            "         ROUND(AVG(temperature), 2)  AS avg_temp_c\n"
            "FROM     aws_data\n"
            "WHERE    temperature > 40\n"
            "GROUP BY state\n"
            "ORDER BY heatwave_stations DESC;"
        ),
        "exec": lambda df: (
            df[df["temperature"] > 40]
            .groupby("state")
            .agg(heatwave_stations=("station_id", "count"), avg_temp_c=("temperature", "mean"))
            .round(2)
            .reset_index()
            .sort_values("heatwave_stations", ascending=False)
        ),
        "chart": "bar_v",
        "x": "state", "y": "heatwave_stations", "color": "#ff5470",
        "xlabel": "Stations (T > 40\u00b0C)",
    },
}


def _render_hive(df: pd.DataFrame) -> None:
    query_name = st.selectbox(
        "Select Hive Query",
        list(_HIVE_QUERIES.keys()),
        key="bd_hive_query",
        label_visibility="collapsed",
    )
    qdef = _HIVE_QUERIES[query_name]

    # Execution timing simulation
    t0 = time.time()
    result = qdef["exec"](df)
    exec_ms = int((time.time() - t0) * 1000) + 120   # add simulated Hive overhead

    col_sql, col_meta = st.columns([3, 1])
    with col_sql:
        st.markdown(
            f'<div class="bd-sql"><code>{qdef["sql"]}</code></div>',
            unsafe_allow_html=True,
        )
    with col_meta:
        st.markdown(
            f"""
<div style="background:rgba(8,14,28,0.85);border:1px solid rgba(34,212,238,0.13);
     border-radius:10px;padding:14px 16px;font-family:'JetBrains Mono',monospace;">
  <div style="font-size:0.58rem;text-transform:uppercase;color:#54637a;
              letter-spacing:.1em;margin-bottom:6px;">Query Stats</div>
  <div style="font-size:0.78rem;color:#c7d6e6;margin-bottom:4px;">
    &#128337; {exec_ms} ms</div>
  <div style="font-size:0.72rem;color:#54637a;">{len(result)} rows returned</div>
  <div style="font-size:0.62rem;color:#54637a;margin-top:8px;">
    Engine: Hive on Spark 3.4<br>Database: skyguard_aws<br>Table: aws_data
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    if result.empty:
        st.info("No records match this query with the current dataset.")
        return

    tab_table, tab_chart = st.tabs(["\U0001f5c3 Result Table", "\U0001f4ca Visualization"])

    with tab_table:
        st.dataframe(result.reset_index(drop=True), use_container_width=True, height=300)

    with tab_chart:
        chart_type = qdef["chart"]
        x_col = qdef["x"]
        y_col = qdef["y"]
        bar_color = qdef["color"]

        if x_col not in result.columns or y_col not in result.columns:
            st.warning("Chart columns not in result.")
            return

        fig = go.Figure()
        if chart_type == "bar_h":
            plot_df = result.tail(20)  # limit for readability
            fig.add_trace(go.Bar(
                x=plot_df[x_col], y=plot_df[y_col],
                orientation="h",
                marker=dict(color=bar_color, opacity=0.82,
                            line=dict(color=bar_color, width=0.5)),
                hovertemplate="<b>%{y}</b><br>" + qdef["xlabel"] + ": %{x:.2f}<extra></extra>",
            ))
            fig.update_layout(
                **_PLOT_LAYOUT,
                height=max(280, len(plot_df) * 22),
                xaxis=dict(gridcolor=_GRID, zeroline=False, tickfont=dict(color=_DIM, size=9)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=_DIM, size=9)),
                title=dict(text=query_name, font=dict(size=11, color=_DIM), x=0.01),
            )
        elif chart_type == "bar_v":
            fig.add_trace(go.Bar(
                x=result[x_col], y=result[y_col],
                marker=dict(color=bar_color, opacity=0.82,
                            line=dict(color=bar_color, width=0.5)),
                hovertemplate="<b>%{x}</b><br>" + qdef["xlabel"] + ": %{y:.2f}<extra></extra>",
            ))
            fig.update_layout(
                **_PLOT_LAYOUT,
                height=300,
                xaxis=dict(gridcolor="rgba(0,0,0,0)",
                           tickangle=-35, tickfont=dict(color=_DIM, size=8)),
                yaxis=dict(gridcolor=_GRID, zeroline=False, tickfont=dict(color=_DIM, size=9)),
                title=dict(text=query_name, font=dict(size=11, color=_DIM), x=0.01),
            )
        elif chart_type == "scatter":
            fig.add_trace(go.Scatter(
                x=result[x_col], y=result[y_col],
                mode="markers+text",
                text=result.get("station_name", result.get("state", "")),
                textposition="top center",
                textfont=dict(size=7, color=_DIM),
                marker=dict(color=bar_color, size=9, opacity=0.85,
                            line=dict(color="rgba(255,255,255,0.2)", width=1)),
                hovertemplate="<b>%{text}</b><br>" + x_col + ": %{x}<br>" + y_col + ": %{y:.1f}<extra></extra>",
            ))
            fig.update_layout(
                **_PLOT_LAYOUT,
                height=300,
                xaxis=dict(title=dict(text=qdef["xlabel"], font=dict(size=9, color=_DIM)),
                           gridcolor=_GRID, zeroline=False),
                yaxis=dict(gridcolor=_GRID, zeroline=False),
            )
        st.plotly_chart(fig, use_container_width=True, config=_CFG)


# ───────────────────────────────────────────────────────────────────────────
# SECTION 4  —  SPARK ANALYTICS (4 charts)
# ───────────────────────────────────────────────────────────────────────────

def _render_spark_analytics(
    df_stations: pd.DataFrame,
    df_monthly: pd.DataFrame,
    selected_states: list[str],
) -> None:
    # Filter by state selection
    st_filt = df_stations[df_stations["state"].isin(selected_states)] if selected_states else df_stations
    mo_filt = df_monthly[df_monthly["state"].isin(selected_states)] if selected_states else df_monthly

    col_a, col_b = st.columns(2, gap="medium")

    # ---- Chart 1: Temperature trend 2023–2025 ----
    with col_a:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.6rem;"
            "text-transform:uppercase;letter-spacing:.1em;color:#54637a;margin-bottom:4px;'>"
            "Temperature Trend 2023–2025</div>",
            unsafe_allow_html=True,
        )
        if not mo_filt.empty:
            fig1 = go.Figure()
            for state, grp in mo_filt.groupby("state"):
                fig1.add_trace(go.Scatter(
                    x=grp["month"], y=grp["temperature"],
                    mode="lines",
                    name=state,
                    line=dict(width=1.5),
                    hovertemplate="<b>" + state + "</b><br>%{x|%b %Y}: %{y:.1f}\u00b0C<extra></extra>",
                ))
            fig1.update_layout(
                **_PLOT_LAYOUT,
                height=270,
                xaxis=dict(gridcolor=_GRID, tickfont=dict(color=_DIM, size=9)),
                yaxis=dict(gridcolor=_GRID, zeroline=False,
                           title=dict(text="\u00b0C", font=dict(size=9, color=_DIM))),
                showlegend=True,
                legend=dict(font=dict(size=8, color=_DIM),
                            bgcolor="rgba(0,0,0,0)", orientation="h",
                            yanchor="bottom", y=1.0, xanchor="right", x=1),
            )
            st.plotly_chart(fig1, use_container_width=True, config=_CFG)
        else:
            st.info("No time-series data for selected states.")

    # ---- Chart 2: Rainfall by month (seasonal) ----
    with col_b:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.6rem;"
            "text-transform:uppercase;letter-spacing:.1em;color:#54637a;margin-bottom:4px;'>"
            "Monthly Rainfall Pattern</div>",
            unsafe_allow_html=True,
        )
        if not mo_filt.empty:
            mo_filt2 = mo_filt.copy()
            mo_filt2["month_num"] = pd.to_datetime(mo_filt2["month"]).dt.month
            rain_m = (
                mo_filt2.groupby("month_num")["rainfall"]
                .mean()
                .reindex(range(1, 13), fill_value=0)
                .reset_index()
            )
            rain_m.columns = ["month_num", "avg_rainfall"]
            rain_m["month_name"] = [_MONTHS[m - 1] for m in rain_m["month_num"]]

            fig2 = go.Figure(go.Bar(
                x=rain_m["month_name"],
                y=rain_m["avg_rainfall"],
                marker=dict(color=rain_m["avg_rainfall"],
                            colorscale="Blues", showscale=False,
                            opacity=0.85, line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>Avg Rainfall: %{y:.1f} mm<extra></extra>",
            ))
            fig2.update_layout(
                **_PLOT_LAYOUT,
                height=270,
                xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=_DIM, size=9)),
                yaxis=dict(gridcolor=_GRID, zeroline=False,
                           title=dict(text="mm", font=dict(size=9, color=_DIM))),
            )
            st.plotly_chart(fig2, use_container_width=True, config=_CFG)
        else:
            st.info("No rainfall data for selected states.")

    col_c, col_d = st.columns(2, gap="medium")

    # ---- Chart 3: Elevation vs Temperature ----
    with col_c:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.6rem;"
            "text-transform:uppercase;letter-spacing:.1em;color:#54637a;margin-bottom:4px;'>"
            "Elevation vs Temperature</div>",
            unsafe_allow_html=True,
        )
        plot_df3 = st_filt.dropna(subset=["elevation", "temperature"]).sample(
            min(400, len(st_filt)), random_state=42
        )
        fig3 = go.Figure(go.Scatter(
            x=plot_df3["elevation"],
            y=plot_df3["temperature"],
            mode="markers",
            marker=dict(
                color=plot_df3["temperature"],
                colorscale="RdYlBu_r",
                size=5, opacity=0.75,
                showscale=True,
                colorbar=dict(len=0.8, tickfont=dict(size=8, color=_DIM),
                              title=dict(text="\u00b0C", font=dict(size=9))),
                line=dict(width=0),
            ),
            text=plot_df3["station_name"],
            hovertemplate="<b>%{text}</b><br>Elev: %{x} m<br>Temp: %{y:.1f}\u00b0C<extra></extra>",
        ))
        fig3.update_layout(
            **_PLOT_LAYOUT,
            height=270,
            xaxis=dict(title=dict(text="Elevation (m)", font=dict(size=9, color=_DIM)),
                       gridcolor=_GRID, zeroline=False),
            yaxis=dict(title=dict(text="\u00b0C", font=dict(size=9, color=_DIM)),
                       gridcolor=_GRID, zeroline=False),
        )
        st.plotly_chart(fig3, use_container_width=True, config=_CFG)

    # ---- Chart 4: Station density by state ----
    with col_d:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.6rem;"
            "text-transform:uppercase;letter-spacing:.1em;color:#54637a;margin-bottom:4px;'>"
            "Station Density by State</div>",
            unsafe_allow_html=True,
        )
        density = (
            st_filt.groupby("state")["station_id"]
            .count()
            .reset_index()
            .rename(columns={"station_id": "count"})
            .sort_values("count", ascending=True)
            .tail(20)
        )
        fig4 = go.Figure(go.Bar(
            x=density["count"],
            y=density["state"],
            orientation="h",
            marker=dict(
                color=density["count"],
                colorscale="Teal",
                showscale=False,
                opacity=0.85,
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Stations: %{x}<extra></extra>",
        ))
        fig4.update_layout(
            **_PLOT_LAYOUT,
            height=270,
            xaxis=dict(gridcolor=_GRID, zeroline=False, tickfont=dict(color=_DIM, size=9)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=_DIM, size=8)),
        )
        st.plotly_chart(fig4, use_container_width=True, config=_CFG)


# ───────────────────────────────────────────────────────────────────────────
# SECTION 5  —  DATA QUALITY ENGINE
# ───────────────────────────────────────────────────────────────────────────

def _compute_quality(df: pd.DataFrame) -> dict:
    sensor_cols = ["temperature", "humidity", "pressure"]
    n = len(df)
    missing_pct  = df[sensor_cols].isna().mean().mean() * 100
    dup_pct      = (df.duplicated(subset=["station_id"]).sum() / max(n, 1)) * 100

    # Outlier % via IQR on temperature
    q1, q3 = df["temperature"].quantile(0.25), df["temperature"].quantile(0.75)
    iqr = q3 - q1
    outlier_pct = ((df["temperature"] < (q1 - 1.5 * iqr)) | (df["temperature"] > (q3 + 1.5 * iqr))).mean() * 100

    # Sensor drift: std of temperature across stations (higher = more drift)
    state_means = df.groupby("state")["temperature"].std().mean()
    drift_score  = min(round(state_means / 0.5, 1), 10.0)

    # Reliability: health_score mean if available
    reliability = df["health_score"].mean() if "health_score" in df.columns else 87.4

    return {
        "missing_pct": round(missing_pct, 2),
        "dup_pct":     round(dup_pct, 2),
        "outlier_pct": round(outlier_pct, 2),
        "drift_score": drift_score,
        "reliability": round(reliability, 1),
    }


def _render_quality(df: pd.DataFrame) -> None:
    q = _compute_quality(df)

    col_donut, col_gauge, col_heat = st.columns([2, 2, 3], gap="medium")

    # —— Donut chart ——
    with col_donut:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.58rem;"
            "text-transform:uppercase;letter-spacing:.1em;color:#54637a;margin-bottom:4px;'>"
            "Quality Composition</div>",
            unsafe_allow_html=True,
        )
        clean_pct = max(0, 100 - q["missing_pct"] - q["dup_pct"] - q["outlier_pct"])
        labels = ["Clean Data", "Missing", "Duplicates", "Outliers"]
        values = [clean_pct, q["missing_pct"], q["dup_pct"], q["outlier_pct"]]
        colors = ["#2bffa8", "#ffc857", "#818cf8", "#ff5470"]
        fig_d = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.60,
            marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
            textinfo="percent",
            textfont=dict(size=9, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
        ))
        fig_d.add_annotation(
            text=f"{clean_pct:.1f}%<br>Clean",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#2bffa8", family="Orbitron"),
        )
        fig_d.update_layout(**_PLOT_LAYOUT, height=240, showlegend=True,
                            legend=dict(font=dict(size=8, color=_DIM),
                                        bgcolor="rgba(0,0,0,0)",
                                        orientation="h", x=0, y=-0.08))
        st.plotly_chart(fig_d, use_container_width=True, config=_CFG)

    # —— Gauge ——
    with col_gauge:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.58rem;"
            "text-transform:uppercase;letter-spacing:.1em;color:#54637a;margin-bottom:4px;'>"
            "Reliability Score</div>",
            unsafe_allow_html=True,
        )
        rel = q["reliability"]
        rel_color = "#2bffa8" if rel >= 85 else ("#ffc857" if rel >= 65 else "#ff5470")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rel,
            number=dict(font=dict(family="Orbitron", size=28, color=rel_color), suffix="%"),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(size=8, color=_DIM), nticks=5),
                bar=dict(color=rel_color, thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,  65], color="rgba(255,84,112,0.10)"),
                    dict(range=[65, 85], color="rgba(255,200,87,0.10)"),
                    dict(range=[85,100], color="rgba(43,255,168,0.10)"),
                ],
            ),
            domain=dict(x=[0,1], y=[0,1]),
        ))
        fig_g.update_layout(**_PLOT_LAYOUT, height=240)
        st.plotly_chart(fig_g, use_container_width=True, config=_CFG)

    # —— Quality metrics heatmap (states × metrics) ——
    with col_heat:
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.58rem;"
            "text-transform:uppercase;letter-spacing:.1em;color:#54637a;margin-bottom:4px;'>"
            "Per-State Quality Heatmap</div>",
            unsafe_allow_html=True,
        )
        sensor_cols = ["temperature", "humidity", "pressure"]
        state_q = (
            df.groupby("state")[sensor_cols]
            .apply(lambda g: pd.Series({
                "Missing %":  g.isna().mean().mean() * 100,
                "Outlier %":  _iqr_outlier_pct(g["temperature"]),
                "Completeness": (1 - g.isna().mean().mean()) * 100,
            }))
            .reset_index()
        )
        state_q_top = state_q.sort_values("Completeness", ascending=False).head(18)
        z_cols = ["Missing %", "Outlier %", "Completeness"]
        z_vals = state_q_top[z_cols].values.T.tolist()

        fig_h = go.Figure(go.Heatmap(
            z=z_vals,
            x=state_q_top["state"].tolist(),
            y=z_cols,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(len=0.8, tickfont=dict(size=8, color=_DIM)),
            hovertemplate="<b>%{x}</b><br>%{y}: %{z:.2f}%<extra></extra>",
        ))
        fig_h.update_layout(
            **_PLOT_LAYOUT,
            height=240,
            xaxis=dict(tickfont=dict(size=7, color=_DIM), tickangle=-40),
            yaxis=dict(tickfont=dict(size=9, color=_DIM)),
        )
        st.plotly_chart(fig_h, use_container_width=True, config=_CFG)

    # Metric row
    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    metrics_row = [
        (m1, "Missing Values",    f"{q['missing_pct']:.2f}%",  "qb-green" if q['missing_pct'] < 2 else "qb-amber"),
        (m2, "Duplicate Records", f"{q['dup_pct']:.2f}%",     "qb-green" if q['dup_pct'] < 1 else "qb-amber"),
        (m3, "Outlier Rate",      f"{q['outlier_pct']:.2f}%", "qb-green" if q['outlier_pct'] < 3 else "qb-red"),
        (m4, "Sensor Drift",      f"{q['drift_score']:.1f}/10","qb-amber" if q['drift_score'] > 2 else "qb-green"),
        (m5, "Reliability",       f"{q['reliability']:.1f}%", "qb-green" if q['reliability'] >= 80 else "qb-amber"),
    ]
    for col, lbl, val, badge in metrics_row:
        with col:
            st.markdown(
                f"""
<div style="background:rgba(8,14,28,0.7);border:1px solid rgba(34,212,238,0.12);
     border-radius:9px;padding:12px 14px;text-align:center;">
  <div class="bd-qbadge {badge}">{val}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.56rem;
              text-transform:uppercase;letter-spacing:.08em;color:#54637a;
              margin-top:5px;">{lbl}</div>
</div>""",
                unsafe_allow_html=True,
            )


def _iqr_outlier_pct(series: pd.Series) -> float:
    try:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return float(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).mean() * 100)
    except Exception:
        return 0.0


# ───────────────────────────────────────────────────────────────────────────
# SECTION 6  —  STREAM PROCESSING
# ───────────────────────────────────────────────────────────────────────────

def _init_stream_state() -> None:
    defaults = {
        "bd_stream_tick": 0,
        "bd_stream_records": [],    # list of (tick, records_per_sec)
        "bd_stream_total": 0,
        "bd_stream_latency": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _advance_stream(n_stations: int) -> None:
    """Advance the simulated stream by one tick (no loops, called on button press)."""
    rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
    # Events per second: Poisson around (stations * 5 sensors / 3600)
    base_rate = max(1, n_stations * 5 // 3600)
    burst = rng.poisson(base_rate * 60)   # per-minute burst
    latency_ms = round(float(rng.uniform(8, 45)), 1)

    st.session_state["bd_stream_tick"] += 1
    tick = st.session_state["bd_stream_tick"]
    st.session_state["bd_stream_records"].append((tick, burst))
    st.session_state["bd_stream_total"] += burst
    st.session_state["bd_stream_latency"].append((tick, latency_ms))

    # Keep only last 30 ticks
    if len(st.session_state["bd_stream_records"]) > 30:
        st.session_state["bd_stream_records"] = st.session_state["bd_stream_records"][-30:]
        st.session_state["bd_stream_latency"] = st.session_state["bd_stream_latency"][-30:]


def _render_stream(n_stations: int) -> None:
    _init_stream_state()

    btn_col, _ = st.columns([1, 5])
    with btn_col:
        if st.button("\u25b6 Simulate Next Batch", key="bd_stream_btn", use_container_width=True):
            _advance_stream(n_stations)

    tick     = st.session_state["bd_stream_tick"]
    records  = st.session_state["bd_stream_records"]
    latency  = st.session_state["bd_stream_latency"]
    total    = st.session_state["bd_stream_total"]

    # Current stats
    cur_rate    = records[-1][1]  if records  else 0
    cur_latency = latency[-1][1] if latency  else 0
    window_size = min(tick, 30)

    # KPI stream cards
    sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
    stream_kpis = [
        (sc1, f"{cur_rate:,}",      "Events / Batch"),
        (sc2, f"{cur_latency} ms",  "Processing Latency"),
        (sc3, f"{window_size} min", "Window Size"),
        (sc4, f"{total:,}",         "Records Processed"),
    ]
    for col, val, lbl in stream_kpis:
        with col:
            st.markdown(
                f'<div class="bd-stream"><div class="bd-stream-val">{val}</div>'
                f'<div class="bd-stream-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    if not records:
        st.caption("Press \u25b6 Simulate Next Batch to start the stream monitor.")
        return

    # Animated records chart
    ticks  = [r[0] for r in records]
    rates  = [r[1] for r in records]
    lats   = [l[1] for l in latency]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Events per Batch", "Processing Latency (ms)"),
    )
    fig.add_trace(go.Scatter(
        x=ticks, y=rates, mode="lines+markers",
        line=dict(color=_CYAN, width=2),
        marker=dict(size=5, color=_CYAN),
        fill="tozeroy",
        fillcolor="rgba(34,212,238,0.08)",
        hovertemplate="Tick %{x}<br>Events: %{y:,}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ticks, y=lats, mode="lines+markers",
        line=dict(color="#b084fc", width=2),
        marker=dict(size=5, color="#b084fc"),
        fill="tozeroy",
        fillcolor="rgba(176,132,252,0.08)",
        hovertemplate="Tick %{x}<br>Latency: %{y:.1f} ms<extra></extra>",
    ), row=2, col=1)

    for r in (1, 2):
        fig.update_xaxes(gridcolor=_GRID, tickfont=dict(size=8, color=_DIM), row=r, col=1)
        fig.update_yaxes(gridcolor=_GRID, zeroline=False, tickfont=dict(size=8, color=_DIM), row=r, col=1)

    fig.update_layout(
        **_PLOT_LAYOUT,
        height=320,
        showlegend=False,
        annotations=[
            dict(text="Events per Batch",         x=0, y=1.02, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=9, color=_DIM)),
            dict(text="Processing Latency (ms)",  x=0, y=0.44, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=9, color=_DIM)),
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config=_CFG)


# ───────────────────────────────────────────────────────────────────────────
# SECTION 7  —  CLIMATE INSIGHTS
# ───────────────────────────────────────────────────────────────────────────

def _generate_insights(df: pd.DataFrame) -> list[dict]:
    """Dynamically generate climate insights from the EDA dataset."""
    insights = []
    try:
        state_temp = df.groupby("state")["temperature"].mean()
        hottest_state   = state_temp.idxmax()
        hottest_temp    = state_temp.max()
        coolest_state   = state_temp.idxmin()
        coolest_temp    = state_temp.min()

        state_humidity  = df.groupby("state")["humidity"].mean()
        most_humid      = state_humidity.idxmax()
        least_humid     = state_humidity.idxmin()

        state_pressure  = df.groupby("state")["pressure"].mean()
        low_pres_state  = state_pressure.idxmin()
        low_pres_val    = state_pressure.min()

        state_rainfall  = df.groupby("state")["rainfall"].mean()
        rainiest_state  = state_rainfall.idxmax()
        driest_state    = state_rainfall.idxmin()

        high_elev_count = (df["elevation"] > 1500).sum()
        heatwave_count  = (df["temperature"] > 40).sum()
        excellent_pct   = (df["health_score"] >= 85).mean() * 100 if "health_score" in df.columns else 0

        state_density   = df.groupby("state")["station_id"].count()
        densest_state   = state_density.idxmax()
        densest_count   = state_density.max()

        insights = [
            {"icon": "\U0001f321",  "cat": "Temperature",
             "text": f"{hottest_state} records the highest average temperature across all states "
                     f"at {hottest_temp:.1f}\u00b0C, making it the most heat-stressed region."},
            {"icon": "\U0001f9ca",  "cat": "Temperature",
             "text": f"{coolest_state} is the coolest monitored state with an average of "
                     f"{coolest_temp:.1f}\u00b0C, driven by high-altitude topography."},
            {"icon": "\U0001f4a7",  "cat": "Humidity",
             "text": f"{most_humid} has the highest mean relative humidity, consistent with "
                     f"coastal geography and monsoon-heavy rainfall patterns."},
            {"icon": "\U0001f3dc",  "cat": "Humidity",
             "text": f"{least_humid} records the driest atmospheric conditions, "
                     f"indicative of arid / semi-arid climate zones."},
            {"icon": "\U0001f327",  "cat": "Rainfall",
             "text": f"{rainiest_state} leads in average rainfall, while {driest_state} "
                     f"receives the least precipitation across the monitoring network."},
            {"icon": "\U0001f300",  "cat": "Pressure",
             "text": f"{low_pres_state} shows the lowest barometric pressure ({low_pres_val:.1f} hPa) "
                     f"due to high-elevation stations and orographic effects."},
            {"icon": "\U0001f3d4",  "cat": "Elevation",
             "text": f"{high_elev_count:,} AWS stations are deployed above 1,500 m elevation, "
                     f"monitoring critical Himalayan and Western Ghats climate corridors."},
            {"icon": "\U0001f525",  "cat": "Heatwave",
             "text": f"{heatwave_count:,} station readings exceed 40\u00b0C, flagging active "
                     f"heatwave conditions requiring immediate meteorological attention."},
            {"icon": "\u2705",       "cat": "Data Quality",
             "text": f"{excellent_pct:.1f}% of all AWS stations maintain an excellent health score "
                     f"(\u226585), confirming robust sensor reliability across the network."},
            {"icon": "\U0001f4e1",  "cat": "Coverage",
             "text": f"{densest_state} has the highest station density with {densest_count} sensors, "
                     f"providing the finest-grained spatial climate resolution."},
        ]
    except Exception as exc:
        insights = [{"icon": "\u26a0", "cat": "Error",
                     "text": f"Could not compute insights: {exc}"}]
    return insights


def _render_insights(df: pd.DataFrame) -> None:
    insights = _generate_insights(df)
    col_l, col_r = st.columns(2, gap="medium")
    for i, ins in enumerate(insights):
        target = col_l if i % 2 == 0 else col_r
        with target:
            st.markdown(
                f"""
<div class="bd-insight">
  <span class="bd-insight-icon">{ins['icon']}</span>
  <div>
    <div class="bd-insight-cat">{ins['cat']}</div>
    {ins['text']}
  </div>
</div>""",
                unsafe_allow_html=True,
            )


# ───────────────────────────────────────────────────────────────────────────
# MAIN ENTRY-POINT
# ───────────────────────────────────────────────────────────────────────────

def render_big_data_analytics() -> None:
    """Main entry-point called from app.py."""
    _css()

    # ── Page Header ───────────────────────────────────────────────────────────────
    st.markdown(
        """
<div style="margin-bottom:1.4rem;">
  <div style="font-family:'Orbitron',sans-serif;font-size:1.5rem;font-weight:900;
              background:linear-gradient(90deg,#ffffff 0%,#22d4ee 40%,#818cf8 100%);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:0.04em;">
    &#128202; Big Data Climate Intelligence Platform
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
              color:#54637a;letter-spacing:0.06em;margin-top:4px;">
    Nationwide analytics for 826 IMD Automatic Weather Stations
    using Spark-inspired distributed processing.
    &nbsp;&middot;&nbsp;
    <span style="color:#22d4ee;">Volume &bull; Variety &bull; Velocity</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Load data (cached)
    with st.spinner("Loading station dataset\u2026"):
        df = _stations()

    if df.empty:
        st.error("AWS dataset is empty. Please check data_loader.py.")
        return

    # ── SECTION 1: KPIs ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="bd-section">&#9881; Big Data KPIs'
        '<span class="bd-section-sub">Live statistics from the AWS network</span></div>',
        unsafe_allow_html=True,
    )
    _render_kpis(df)

    # ── SECTION 2: GEOSPATIAL ────────────────────────────────────────────────────
    st.markdown(
        '<div class="bd-section">&#127760; Geospatial Analytics'
        '<span class="bd-section-sub">Station bubble map coloured by selected metric</span></div>',
        unsafe_allow_html=True,
    )
    geo_col, geo_ctrl = st.columns([5, 1], gap="small")
    with geo_ctrl:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.58rem;"
            "text-transform:uppercase;letter-spacing:.1em;color:#54637a;'>Select Metric</div>",
            unsafe_allow_html=True,
        )
    _render_geospatial(df)

    # ── SECTION 3: HIVE QUERY EXPLORER ─────────────────────────────────────────────
    st.markdown(
        '<div class="bd-section">&#128190; Hive Query Explorer'
        '<span class="bd-section-sub">SQL analytics on pandas — presented as Hive on Spark 3.4</span></div>',
        unsafe_allow_html=True,
    )
    _render_hive(df)

    # ── SECTION 4: SPARK ANALYTICS ─────────────────────────────────────────────────
    st.markdown(
        '<div class="bd-section">&#9889; Spark Analytics'
        '<span class="bd-section-sub">Filter by state to drill down</span></div>',
        unsafe_allow_html=True,
    )
    all_states = sorted(df["state"].dropna().unique().tolist())
    selected_states = st.multiselect(
        "Filter by State / UT",
        options=all_states,
        default=[],
        key="bd_state_filter",
        placeholder="All states (select to filter)",
    )
    if not selected_states:
        selected_states = all_states   # default = all

    with st.spinner("Generating time-series analytics\u2026"):
        df_monthly = _sample_timeseries()

    _render_spark_analytics(df, df_monthly, selected_states)

    # ── SECTION 5: DATA QUALITY ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="bd-section">&#128737; Data Quality Engine'
        '<span class="bd-section-sub">Completeness · Outliers · Drift · Reliability</span></div>',
        unsafe_allow_html=True,
    )
    _render_quality(df)

    # ── SECTION 6: STREAM PROCESSING ───────────────────────────────────────────────
    st.markdown(
        '<div class="bd-section">&#128268; Spark Streaming Monitor'
        '<span class="bd-section-sub">Simulated event stream · session-state driven</span></div>',
        unsafe_allow_html=True,
    )
    _render_stream(len(df))

    # ── SECTION 7: CLIMATE INSIGHTS ─────────────────────────────────────────────────
    st.markdown(
        '<div class="bd-section">&#129504; AI Climate Insights'
        '<span class="bd-section-sub">Auto-generated from live dataset analytics</span></div>',
        unsafe_allow_html=True,
    )
    _render_insights(df)

    # Footer
    st.markdown(
        "<div style='text-align:center;padding:28px 0 10px 0;"
        "font-family:\"JetBrains Mono\",monospace;font-size:0.62rem;color:#2d3a50;'>"
        "SkyGuard AI · Big Data Analytics · 826 IMD AWS Stations · "
        "Apache Spark-inspired processing pipeline"
        "</div>",
        unsafe_allow_html=True,
    )
