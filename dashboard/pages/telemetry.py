"""
================================================================================
dashboard/pages/telemetry.py — SkyGuard AI Live Telemetry & Imputation
================================================================================
Renders the raw-vs-ground-truth-vs-imputed telemetry inspector: station /
sensor / time-range controls, a Plotly time-series overlay, and performance
metrics (Raw MAE, Imputed MAE, Error Reduction, Confidence) computed from
`MeteorologicalImputer.impute_dataframe()`.

Data policy: every plotted series and every metric is derived from
`df_sample` (the real telemetry slice passed in) run through the real
imputer. Nothing is simulated — if a metric can't be computed from the
available columns for the current selection (e.g. no clean-truth column, or
no degraded points in the selected window), the UI shows an explicit
"N/A" / empty-state rather than a placeholder number.

Requires `dashboard.styles.inject_css()` to have been called earlier in the
page so shared tokens are available; this module injects a small amount of
additional, page-specific CSS on top of that shared theme.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_SENSOR_OPTIONS = {
    "temperature_c": "Temperature (°C)",
    "air_pressure_mbar": "Surface Pressure (mbar)",
    "relative_humidity_pct": "Relative Humidity (%)",
}

_ACCENT_RAW = "#ff5470"
_ACCENT_IMPUTED = "#2bffa8"
_ACCENT_TRUTH = "rgba(148, 163, 184, 0.55)"


# ==========================================================================
# CSS
# ==========================================================================
def _inject_telemetry_css() -> None:
    st.markdown(
        """
        <style>
        .sg-tele-title{
            font-family:'Orbitron', sans-serif;
            font-size: 1.3rem; font-weight:700;
            color:#e7f6ff; letter-spacing:0.03em;
        }
        .sg-tele-caption{
            font-family:'JetBrains Mono', monospace;
            font-size: 0.72rem; color:#8aa2bd;
            letter-spacing:0.04em; margin-bottom: 14px;
        }
        .sg-tele-panel{
            background: linear-gradient(160deg, rgba(16,24,42,0.6), rgba(8,13,24,0.5));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            backdrop-filter: blur(18px) saturate(150%);
            -webkit-backdrop-filter: blur(18px) saturate(150%);
            box-shadow: 0 8px 28px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
            padding: 14px 18px 16px 18px;
            margin-bottom: 14px;
        }
        .sg-tele-metric{
            position: relative;
            border-radius: 14px;
            padding: 12px 14px 10px 14px;
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 3px solid var(--sg-m-color, #22e8ff);
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
            min-height: 92px;
        }
        .sg-tele-metric:hover{
            transform: translateY(-3px);
            border-color: var(--sg-m-color, #22e8ff);
            box-shadow: 0 10px 26px rgba(0,0,0,0.4), 0 0 22px 0 var(--sg-m-shadow, rgba(34,232,255,0.25));
        }
        .sg-tele-metric-label{
            font-family:'JetBrains Mono', monospace;
            font-size: 0.66rem; font-weight:600;
            letter-spacing:0.08em; text-transform:uppercase;
            color:#8aa2bd;
        }
        .sg-tele-metric-value{
            font-family:'Orbitron', sans-serif;
            font-weight:800; font-size: 1.55rem;
            color:#f4fbff; margin-top: 8px;
            text-shadow: 0 0 18px var(--sg-m-shadow, rgba(34,232,255,0.25));
        }
        .sg-tele-metric-value .sg-tele-unit{
            font-size: 0.85rem; font-weight:600; color:#8aa2bd; margin-left: 3px;
        }
        .sg-tele-metric-sub{
            font-family:'Rajdhani', sans-serif;
            font-size: 0.72rem; color:#54637a; margin-top: 3px;
        }
        .sg-tele-empty{
            text-align:center; padding: 28px 10px; color:#54637a;
            font-family:'JetBrains Mono', monospace; font-size: 0.78rem; letter-spacing:0.04em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(col, *, label: str, value: str, unit: str, sub: str, color: str) -> None:
    with col:
        st.markdown(
            f"""
            <div class="sg-tele-metric" style="--sg-m-color:{color}; --sg-m-shadow:{color}55;">
                <div class="sg-tele-metric-label">{label}</div>
                <div class="sg-tele-metric-value">{value}<span class="sg-tele-unit">{unit}</span></div>
                <div class="sg-tele-metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================================================
# Imputer invocation
# ==========================================================================
def _run_imputer(stn_slice: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Run the real `MeteorologicalImputer` on the given station slice.

    Returns None (and surfaces an `st.error`) if the imputer module can't be
    imported or raises — this page never substitutes fake imputed values.
    """
    try:
        from imputation.correction import MeteorologicalImputer
    except ImportError as exc:
        st.error(f"Could not import `MeteorologicalImputer` from `imputation.correction`: {exc}")
        return None

    try:
        imputer = MeteorologicalImputer(k_neighbors=2)
        return imputer.impute_dataframe(stn_slice)
    except Exception as exc:  # surface real imputer failures, don't mask them
        st.error(f"MeteorologicalImputer.impute_dataframe() raised an error: {exc}")
        return None


# ==========================================================================
# Public entrypoint
# ==========================================================================
def render_live_telemetry(df_sample: pd.DataFrame) -> None:
    """Render the SkyGuard AI Live Telemetry & Imputation inspector page.

    Args:
        df_sample: Real telemetry dataframe, expected to contain at least
            ``station_id``, ``timestamp``, and the raw sensor columns
            (``temperature_c``, ``air_pressure_mbar``,
            ``relative_humidity_pct``). Optional ``clean_<var>`` ground-truth
            columns, if present after imputation, enable the MAE / error
            reduction / confidence metrics. No values are fabricated: if a
            column the metrics need isn't available, that metric is shown as
            "N/A" rather than estimated.

    Controls exposed: Station, Sensor (Meteorological Parameter), and Time
    Range (bounded to the real min/max timestamps of the selected station's
    data).

    Plots: Raw (corrupted observation), Ground Truth (clean target, when
    available), and Imputed (SkyGuard self-healing estimate).
    """
    _inject_telemetry_css()

    st.markdown('<div class="sg-tele-title">📈 Live Telemetry & Imputation Inspector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sg-tele-caption">RAW SENSOR STREAM · GROUND TRUTH · SKYGUARD AI SELF-HEALING IMPUTATION</div>',
        unsafe_allow_html=True,
    )

    if df_sample is None or len(df_sample) == 0:
        st.markdown('<div class="sg-tele-panel"><div class="sg-tele-empty">NO TELEMETRY DATA LOADED</div></div>', unsafe_allow_html=True)
        return

    if "station_id" not in df_sample.columns or "timestamp" not in df_sample.columns:
        st.error("`df_sample` must contain `station_id` and `timestamp` columns.")
        return

    stations = sorted(df_sample["station_id"].dropna().unique().tolist())
    if not stations:
        st.markdown('<div class="sg-tele-panel"><div class="sg-tele-empty">NO STATIONS FOUND IN TELEMETRY DATA</div></div>', unsafe_allow_html=True)
        return

    available_sensors = [c for c in _SENSOR_OPTIONS if c in df_sample.columns]
    if not available_sensors:
        st.error("None of the expected sensor columns were found in `df_sample`.")
        return

    # ---- Controls: Station · Sensor · Time Range ---------------------------
    st.markdown('<div class="sg-tele-panel">', unsafe_allow_html=True)
    c_station, c_sensor, c_range = st.columns([1, 1.3, 2])

    with c_station:
        sel_station = st.selectbox("Station", stations, key="sg_tele_station")

    with c_sensor:
        sel_var = st.selectbox(
            "Sensor",
            available_sensors,
            format_func=lambda v: _SENSOR_OPTIONS.get(v, v),
            key="sg_tele_sensor",
        )

    station_df = df_sample[df_sample["station_id"] == sel_station].copy()
    station_df["timestamp"] = pd.to_datetime(station_df["timestamp"])
    station_df = station_df.sort_values("timestamp")

    with c_range:
        if len(station_df) > 0:
            ts_min = station_df["timestamp"].min().to_pydatetime()
            ts_max = station_df["timestamp"].max().to_pydatetime()
            if ts_min == ts_max:
                st.caption(f"Time Range: single observation at {ts_min}")
                range_start, range_end = ts_min, ts_max
            else:
                range_start, range_end = st.slider(
                    "Time Range",
                    min_value=ts_min,
                    max_value=ts_max,
                    value=(ts_min, ts_max),
                    key="sg_tele_range",
                )
        else:
            range_start, range_end = None, None

    st.markdown("</div>", unsafe_allow_html=True)

    if range_start is None:
        st.markdown(f'<div class="sg-tele-panel"><div class="sg-tele-empty">NO OBSERVATIONS FOR STATION {sel_station}</div></div>', unsafe_allow_html=True)
        return

    stn_slice = station_df[(station_df["timestamp"] >= range_start) & (station_df["timestamp"] <= range_end)].copy()

    if len(stn_slice) == 0:
        st.markdown('<div class="sg-tele-panel"><div class="sg-tele-empty">NO OBSERVATIONS IN THE SELECTED TIME RANGE</div></div>', unsafe_allow_html=True)
        return

    # ---- Run the real imputer ------------------------------------------
    stn_imputed = _run_imputer(stn_slice)
    if stn_imputed is None:
        return

    imputed_col = f"imputed_{sel_var}"
    clean_col = f"clean_{sel_var}"

    if imputed_col not in stn_imputed.columns:
        st.error(f"`MeteorologicalImputer.impute_dataframe()` did not return the expected `{imputed_col}` column.")
        return

    time_x = stn_imputed["timestamp"]
    raw_y = stn_imputed[sel_var]
    imp_y = stn_imputed[imputed_col]
    clean_y = stn_imputed[clean_col] if clean_col in stn_imputed.columns else None

    # ---- Plot: Raw vs Ground Truth vs Imputed ---------------------------
    st.markdown('<div class="sg-tele-panel">', unsafe_allow_html=True)
    fig = go.Figure()

    if clean_y is not None:
        fig.add_trace(
            go.Scatter(
                x=time_x, y=clean_y, mode="lines", name="Ground Truth",
                line=dict(color=_ACCENT_TRUTH, width=2, dash="dot"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=time_x, y=raw_y, mode="lines+markers", name="Raw",
            line=dict(color=_ACCENT_RAW, width=1.6),
            marker=dict(size=4, color=_ACCENT_RAW),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_x, y=imp_y, mode="lines", name="Imputed",
            line=dict(color=_ACCENT_IMPUTED, width=2.6),
        )
    )

    fig.update_layout(
        title=dict(text=f"{sel_station} — {_SENSOR_OPTIONS.get(sel_var, sel_var)}", font=dict(color="#e7f6ff", family="Orbitron, sans-serif", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c7d6e6", family="Rajdhani, sans-serif"),
        xaxis=dict(title="Timestamp", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(title=_SENSOR_OPTIONS.get(sel_var, sel_var), gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#c7d6e6")),
        margin=dict(l=10, r=10, t=52, b=10),
        height=440,
        hoverlabel=dict(bgcolor="rgba(6,10,20,0.95)", bordercolor="rgba(34,232,255,0.45)", font=dict(color="#e7f6ff")),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="sg_telemetry_fig")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Metrics: Raw MAE · Imputed MAE · Error Reduction · Confidence ----
    st.markdown('<div class="sg-tele-panel">', unsafe_allow_html=True)

    if clean_y is None:
        st.markdown(
            '<div class="sg-tele-empty">NO GROUND-TRUTH COLUMN '
            f'(`{clean_col}`) RETURNED — MAE / ERROR REDUCTION CANNOT BE COMPUTED</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    flag_col = "imputation_flag"
    if flag_col in stn_imputed.columns:
        anom_mask = stn_imputed[flag_col] != "RAW_PASSTHROUGH"
    else:
        # No flag column returned — fall back to comparing every point
        # rather than guessing which points were corrected.
        anom_mask = pd.Series(True, index=stn_imputed.index)

    if anom_mask.sum() == 0:
        st.markdown(
            '<div class="sg-tele-empty">NO DEGRADED / IMPUTED OBSERVATIONS IN THIS WINDOW — FLEET NOMINAL</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    raw_errs = np.abs(raw_y[anom_mask] - clean_y[anom_mask])
    imp_errs = np.abs(imp_y[anom_mask] - clean_y[anom_mask])
    raw_mae = float(np.nanmean(raw_errs)) if raw_errs.notna().any() else None
    imp_mae = float(np.nanmean(imp_errs)) if imp_errs.notna().any() else None

    err_red = None
    if raw_mae is not None and imp_mae is not None and raw_mae > 0:
        err_red = max(0.0, (raw_mae - imp_mae) / raw_mae * 100.0)

    confidence_val = None
    for cand_col in ("imputation_confidence", "confidence"):
        if cand_col in stn_imputed.columns:
            conf_series = stn_imputed.loc[anom_mask, cand_col].dropna()
            if len(conf_series) > 0:
                confidence_val = float(conf_series.mean())
            break

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(
        m1, label="Raw MAE",
        value=f"{raw_mae:.2f}" if raw_mae is not None else "N/A",
        unit="", sub=f"{int(anom_mask.sum())} degraded points",
        color=_ACCENT_RAW,
    )
    _metric_card(
        m2, label="Imputed MAE",
        value=f"{imp_mae:.2f}" if imp_mae is not None else "N/A",
        unit="", sub="vs. ground truth",
        color=_ACCENT_IMPUTED,
    )
    _metric_card(
        m3, label="Error Reduction",
        value=f"{err_red:.1f}" if err_red is not None else "N/A",
        unit="%" if err_red is not None else "",
        sub="Raw → Imputed MAE",
        color="#7c8cff",
    )
    _metric_card(
        m4, label="Confidence",
        value=f"{confidence_val * 100:.1f}" if confidence_val is not None and confidence_val <= 1 else (f"{confidence_val:.1f}" if confidence_val is not None else "N/A"),
        unit="%" if confidence_val is not None else "",
        sub="Mean imputer confidence" if confidence_val is not None else "Not returned by imputer",
        color="#b084fc",
    )
    st.markdown("</div>", unsafe_allow_html=True)
