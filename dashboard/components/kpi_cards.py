"""
================================================================================
dashboard/components/kpi_cards.py — SkyGuard AI Command Center KPI Strip
================================================================================
Renders the top-row fleet KPI strip as five premium glassmorphic cards, each
with a hover-lift/glow animation and an embedded Plotly sparkline trend.

Cards:
    1. Total AWS            — active weather stations vs. fleet size
    2. Active Alerts         — current high-severity diagnostic alerts
    3. Fleet Health           — mean prognostic health score (0-100)
    4. Imputation Accuracy   — self-healing imputation accuracy (%)
    5. Hybrid AI F1           — hybrid detector F1 score

Requires `dashboard.styles.inject_css()` to have been called earlier in the
page so shared tokens (--sg-cyan, --sg-panel, etc.) and the `.sg-glass`
primitive are available; this module injects a small amount of additional,
KPI-specific CSS on top of that shared theme.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Card visual theme — one accent per KPI
# --------------------------------------------------------------------------
_ACCENTS = {
    "total_aws": {"hex": "#22e8ff", "rgb": "34, 232, 255"},
    "alerts": {"hex": "#ff5470", "rgb": "255, 84, 112"},
    "fleet_health": {"hex": "#2bffa8", "rgb": "43, 255, 168"},
    "imputation": {"hex": "#7c8cff", "rgb": "124, 140, 255"},
    "f1": {"hex": "#b084fc", "rgb": "176, 132, 252"},
}


def _inject_kpi_css() -> None:
    """Inject the KPI-card-specific glassmorphism + hover-animation CSS."""
    st.markdown(
        """
        <style>
        .sg-kpi-card{
            position: relative;
            border-radius: 18px;
            padding: 16px 18px 10px 18px;
            background: linear-gradient(160deg, rgba(16,24,42,0.65), rgba(8,13,24,0.55));
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(20px) saturate(150%);
            -webkit-backdrop-filter: blur(20px) saturate(150%);
            box-shadow: 0 8px 28px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.05);
            overflow: hidden;
            transition: transform 0.28s cubic-bezier(.2,.8,.2,1),
                        box-shadow 0.28s ease,
                        border-color 0.28s ease;
            min-height: 168px;
            display: flex;
            flex-direction: column;
        }
        .sg-kpi-card::before{
            content:"";
            position:absolute; inset:0;
            background: radial-gradient(120px 80px at 85% -10%, var(--sg-kpi-glow, rgba(34,232,255,0.18)), transparent 70%);
            opacity: 0.9;
            pointer-events: none;
        }
        .sg-kpi-card::after{
            content:"";
            position:absolute;
            left:0; right:0; bottom:0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--sg-kpi-accent, #22e8ff), transparent);
            opacity: 0.55;
            transition: opacity 0.28s ease;
        }
        .sg-kpi-card:hover{
            transform: translateY(-5px) scale(1.012);
            border-color: var(--sg-kpi-accent, #22e8ff);
            box-shadow:
                0 16px 40px rgba(0,0,0,0.5),
                0 0 0 1px rgba(255,255,255,0.03) inset,
                0 0 28px 0 var(--sg-kpi-shadow, rgba(34,232,255,0.30));
        }
        .sg-kpi-card:hover::after{ opacity: 1; }

        .sg-kpi-top{
            display:flex; align-items:flex-start; justify-content:space-between;
            gap: 8px;
            position: relative; z-index: 1;
        }
        .sg-kpi-icon{
            width: 34px; height: 34px;
            border-radius: 10px;
            display:flex; align-items:center; justify-content:center;
            font-size: 1.05rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 0 14px 0 var(--sg-kpi-shadow, rgba(34,232,255,0.22));
            flex-shrink: 0;
            transition: transform 0.28s ease, box-shadow 0.28s ease;
        }
        .sg-kpi-card:hover .sg-kpi-icon{
            transform: rotate(-6deg) scale(1.08);
            box-shadow: 0 0 22px 2px var(--sg-kpi-shadow, rgba(34,232,255,0.4));
        }
        .sg-kpi-label{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8aa2bd;
            margin-top: 2px;
        }
        .sg-kpi-delta{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 999px;
            white-space: nowrap;
            letter-spacing: 0.03em;
        }
        .sg-kpi-delta-up{ color:#2bffa8; background: rgba(43,255,168,0.10); border: 1px solid rgba(43,255,168,0.3); }
        .sg-kpi-delta-down{ color:#ff5470; background: rgba(255,84,112,0.10); border: 1px solid rgba(255,84,112,0.3); }
        .sg-kpi-delta-flat{ color:#8aa2bd; background: rgba(138,162,189,0.08); border: 1px solid rgba(138,162,189,0.25); }

        .sg-kpi-value{
            font-family: 'Orbitron', sans-serif;
            font-weight: 800;
            font-size: 1.9rem;
            color: #f4fbff;
            margin-top: 10px;
            line-height: 1.1;
            position: relative; z-index: 1;
            text-shadow: 0 0 22px var(--sg-kpi-shadow, rgba(34,232,255,0.28));
        }
        .sg-kpi-value .sg-kpi-unit{
            font-size: 0.95rem;
            font-weight: 600;
            color: #8aa2bd;
            margin-left: 4px;
        }
        .sg-kpi-sub{
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.78rem;
            color: #54637a;
            margin-top: 2px;
            position: relative; z-index: 1;
        }
        .sg-kpi-spark{
            margin-top: auto;
            padding-top: 6px;
            position: relative; z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _synth_trend(seed: int, base: float, spread: float, n: int = 24) -> List[float]:
    """Generate a smooth, deterministic pseudo-trend around `base`.

    Used only as a graceful fallback sparkline when no historical series is
    present in `demo_data`, so the card never renders empty. Deterministic
    per `seed` so the shape is stable across Streamlit reruns.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=spread * 0.18, size=n)
    walk = np.cumsum(steps)
    walk -= walk.mean()
    trend = base + walk
    return [float(v) for v in trend]


def _delta_badge(values: Sequence[float]) -> str:
    """Return HTML for an up/down/flat delta badge comparing first vs last point."""
    if not values or len(values) < 2:
        return '<span class="sg-kpi-delta sg-kpi-delta-flat">— 0.0%</span>'
    first, last = values[0], values[-1]
    if first == 0:
        pct = 0.0
    else:
        pct = (last - first) / abs(first) * 100.0
    if pct > 0.15:
        return f'<span class="sg-kpi-delta sg-kpi-delta-up">▲ {pct:.1f}%</span>'
    if pct < -0.15:
        return f'<span class="sg-kpi-delta sg-kpi-delta-down">▼ {abs(pct):.1f}%</span>'
    return f'<span class="sg-kpi-delta sg-kpi-delta-flat">— {pct:.1f}%</span>'


def _sparkline_figure(values: Sequence[float], color_hex: str) -> go.Figure:
    """Build a minimal, axis-free Plotly sparkline for embedding in a KPI card."""
    x = list(range(len(values)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=values,
            mode="lines",
            line=dict(color=color_hex, width=2.2, shape="spline", smoothing=0.9),
            fill="tozeroy",
            fillcolor=color_hex.replace(")", ", 0.16)").replace("rgb", "rgba")
            if color_hex.startswith("rgb")
            else _hex_to_rgba(color_hex, 0.16),
            hoverinfo="skip",
        )
    )
    # Marker on the final point to draw the eye to "now"
    fig.add_trace(
        go.Scatter(
            x=[x[-1]] if x else [],
            y=[values[-1]] if values else [],
            mode="markers",
            marker=dict(color=color_hex, size=6, line=dict(color="rgba(255,255,255,0.6)", width=1)),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=2, b=0),
        height=52,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        hovermode=False,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _render_card(
    col,
    *,
    key: str,
    icon: str,
    label: str,
    value_text: str,
    unit: str,
    sub_text: str,
    trend: List[float],
    chart_key: str,
) -> None:
    """Render a single KPI glass card (header + value + sparkline) into `col`."""
    accent = _ACCENTS[key]
    with col:
        st.markdown(
            f"""
            <div class="sg-kpi-card" style="--sg-kpi-accent: {accent['hex']};
                        --sg-kpi-shadow: rgba({accent['rgb']}, 0.30);
                        --sg-kpi-glow: rgba({accent['rgb']}, 0.18);">
                <div class="sg-kpi-top">
                    <div>
                        <div class="sg-kpi-icon">{icon}</div>
                        <div class="sg-kpi-label">{label}</div>
                    </div>
                    {_delta_badge(trend)}
                </div>
                <div class="sg-kpi-value">{value_text}<span class="sg-kpi-unit">{unit}</span></div>
                <div class="sg-kpi-sub">{sub_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sg-kpi-spark">', unsafe_allow_html=True)
        fig = _sparkline_figure(trend, accent["hex"])
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False, "staticPlot": True},
            key=chart_key,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_kpi_cards(demo_data: Dict[str, Any], station_metadata: Dict[str, Any]) -> None:
    """Render the five-card SkyGuard AI fleet KPI strip.

    Args:
        demo_data: Pipeline / demo results dict (e.g. loaded from
            ``artifacts/pipeline_demo_results.json``). Recognized optional
            keys — all are read defensively with safe fallbacks, so this
            renders sensibly even if `demo_data` is ``{}``:
                - "active_stations" (int)
                - "anomalies_detected" (int)
                - "sample_diagnostic_cards" (list)
                - "station_health_summaries" (list of {"overall_score": float, ...})
                - "imputation_accuracy_pct" (float, 0-100)
                - "hybrid_ai_f1" (float, 0-1) or nested under
                  "model_metrics": {"f1_score": float}
                - "trends": {"total_aws" | "alerts" | "fleet_health" |
                  "imputation" | "f1": list[float]} — optional historical
                  series; synthetic sparklines are generated when absent.
        station_metadata: Station registry dict, e.g. ``STATION_METADATA``
            from app.py (station_id -> {"city", "lat", "lon", "elev"}).
    """
    _inject_kpi_css()

    demo_data = demo_data or {}
    station_metadata = station_metadata or {}
    trends = demo_data.get("trends", {}) if isinstance(demo_data.get("trends", {}), dict) else {}

    # ---- 1. Total AWS ----------------------------------------------------
    total_fleet = len(station_metadata) or int(demo_data.get("fleet_size", 10))
    active_stations = int(demo_data.get("active_stations", total_fleet))
    total_trend = trends.get("total_aws") or _synth_trend(1, base=active_stations, spread=max(0.6, total_fleet * 0.05))

    # ---- 2. Active Alerts --------------------------------------------------
    active_alerts = int(
        demo_data.get(
            "active_alerts",
            len(demo_data.get("sample_diagnostic_cards", []) or []) or demo_data.get("anomalies_detected", 0),
        )
    )
    alerts_trend = trends.get("alerts") or _synth_trend(2, base=max(active_alerts, 1), spread=max(1.0, active_alerts * 0.3))

    # ---- 3. Fleet Health ----------------------------------------------------
    summaries = demo_data.get("station_health_summaries", []) or []
    if summaries:
        fleet_health = float(np.mean([s.get("overall_score", 0.0) for s in summaries]))
    else:
        fleet_health = float(demo_data.get("fleet_health_score", 74.5))
    health_trend = trends.get("fleet_health") or _synth_trend(3, base=fleet_health, spread=4.0)
    health_status = (
        "EXCELLENT" if fleet_health >= 90 else "GOOD" if fleet_health >= 75 else "DEGRADING" if fleet_health >= 55 else "CRITICAL"
    )

    # ---- 4. Imputation Accuracy ----------------------------------------------
    imputation_acc = float(demo_data.get("imputation_accuracy_pct", demo_data.get("imputation_accuracy", 94.2)))
    if imputation_acc <= 1.0:  # tolerate a 0-1 fraction being passed in
        imputation_acc *= 100.0
    imputation_trend = trends.get("imputation") or _synth_trend(4, base=imputation_acc, spread=1.6)

    # ---- 5. Hybrid AI F1 -------------------------------------------------
    f1_score = demo_data.get("hybrid_ai_f1")
    if f1_score is None:
        f1_score = demo_data.get("model_metrics", {}).get("f1_score", 0.912)
    f1_score = float(f1_score)
    f1_trend = trends.get("f1") or _synth_trend(5, base=f1_score, spread=0.03)

    cols = st.columns(5, gap="small")

    _render_card(
        cols[0],
        key="total_aws",
        icon="🛰️",
        label="Total AWS",
        value_text=f"{active_stations}/{total_fleet}",
        unit="",
        sub_text="Automatic Weather Stations online",
        trend=total_trend,
        chart_key="sg_kpi_spark_total_aws",
    )
    _render_card(
        cols[1],
        key="alerts",
        icon="🚨",
        label="Active Alerts",
        value_text=f"{active_alerts}",
        unit="",
        sub_text="High-severity diagnostic alerts",
        trend=alerts_trend,
        chart_key="sg_kpi_spark_alerts",
    )
    _render_card(
        cols[2],
        key="fleet_health",
        icon="💚",
        label="Fleet Health",
        value_text=f"{fleet_health:.1f}",
        unit="/100",
        sub_text=f"Prognostic score · {health_status}",
        trend=health_trend,
        chart_key="sg_kpi_spark_fleet_health",
    )
    _render_card(
        cols[3],
        key="imputation",
        icon="🔄",
        label="Imputation Accuracy",
        value_text=f"{imputation_acc:.1f}",
        unit="%",
        sub_text="Self-healing reconstruction accuracy",
        trend=imputation_trend,
        chart_key="sg_kpi_spark_imputation",
    )
    _render_card(
        cols[4],
        key="f1",
        icon="🧠",
        label="Hybrid AI F1",
        value_text=f"{f1_score:.3f}",
        unit="",
        sub_text="Fusion anomaly-detector F1 score",
        trend=f1_trend,
        chart_key="sg_kpi_spark_f1",
    )
