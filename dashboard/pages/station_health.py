"""
================================================================================
dashboard/pages/station_health.py — SkyGuard AI Prognostic Health & RUL
================================================================================
Renders per-station prognostic health as premium, responsive glass cards:
a radial (donut gauge) overall health score, per-sensor score bars, and
Remaining Useful Life (RUL) indicators.

Data policy: reads only from `demo_data["station_health_summaries"]` joined
against the real `station_metadata` registry for city names. A station
present in `station_metadata` but absent from `demo_data` is still listed
(using real metadata) but explicitly marked "NO HEALTH DATA" rather than
given a fabricated score; a sensor with no `days_to_critical` value is
labeled "Stable" rather than assigned an invented RUL.

Requires `dashboard.styles.inject_css()` to have been called earlier in the
page so shared tokens are available; this module injects a small amount of
additional, page-specific CSS on top of that shared theme.
"""

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

_STATUS_COLORS: Dict[str, str] = {
    "EXCELLENT": "#2bffa8",
    "GOOD": "#22e8ff",
    "DEGRADING": "#ffc857",
    "CRITICAL": "#ff5470",
    "NO DATA": "#54637a",
}
_STATUS_ORDER: List[str] = ["EXCELLENT", "GOOD", "DEGRADING", "CRITICAL", "NO DATA"]


# ==========================================================================
# CSS
# ==========================================================================
def _inject_health_css() -> None:
    st.markdown(
        """
        <style>
        .sg-health-title{
            font-family:'Orbitron', sans-serif; font-size: 1.3rem; font-weight:700;
            color:#e7f6ff; letter-spacing:0.03em;
        }
        .sg-health-caption{
            font-family:'JetBrains Mono', monospace; font-size: 0.72rem;
            color:#8aa2bd; letter-spacing:0.04em; margin-bottom: 4px;
        }
        .sg-health-legend{ display:flex; flex-wrap:wrap; gap:10px; margin: 10px 0 16px 0; }
        .sg-health-chip{
            display:flex; align-items:center; gap:6px;
            font-family:'JetBrains Mono', monospace; font-size: 0.68rem; color:#c7d6e6;
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 999px; padding: 4px 10px;
        }
        .sg-health-dot{ width:9px; height:9px; border-radius:50%; box-shadow: 0 0 8px currentColor; }

        .sg-health-card{
            border-radius: 18px;
            background: linear-gradient(160deg, rgba(16,24,42,0.62), rgba(8,13,24,0.52));
            border: 1px solid rgba(255,255,255,0.08);
            border-top: 3px solid var(--sg-h-color, #22e8ff);
            backdrop-filter: blur(18px) saturate(150%);
            -webkit-backdrop-filter: blur(18px) saturate(150%);
            box-shadow: 0 8px 26px rgba(0,0,0,0.35);
            padding: 14px 16px 16px 16px;
            margin-bottom: 18px;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .sg-health-card:hover{
            transform: translateY(-4px);
            box-shadow: 0 14px 36px rgba(0,0,0,0.45), 0 0 26px 0 var(--sg-h-shadow, rgba(34,232,255,0.22));
        }
        .sg-health-card-head{
            display:flex; align-items:flex-start; justify-content:space-between; gap:8px;
        }
        .sg-health-station{
            font-family:'JetBrains Mono', monospace; font-weight:700; font-size: 0.82rem;
            color:#e7f6ff; letter-spacing:0.02em;
        }
        .sg-health-city{
            font-family:'Rajdhani', sans-serif; font-size: 0.82rem; color:#8aa2bd; margin-top: 1px;
        }
        .sg-health-status-badge{
            font-family:'JetBrains Mono', monospace; font-size: 0.64rem; font-weight:700;
            letter-spacing:0.06em; padding: 3px 9px; border-radius:999px;
            color: var(--sg-h-color, #22e8ff);
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--sg-h-color, #22e8ff);
            white-space: nowrap;
        }
        .sg-health-urgency{
            font-family:'JetBrains Mono', monospace; font-size: 0.64rem; color:#54637a;
            margin-top: 4px; letter-spacing: 0.04em;
        }

        .sg-sensor-row{ margin-top: 10px; }
        .sg-sensor-label-row{
            display:flex; justify-content:space-between; align-items:baseline;
            font-family:'JetBrains Mono', monospace; font-size: 0.66rem; color:#8aa2bd;
            letter-spacing: 0.03em; margin-bottom: 3px;
        }
        .sg-sensor-label-row b{ color:#c7d6e6; }
        .sg-sensor-bar-track{
            width:100%; height:6px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow:hidden;
        }
        .sg-sensor-bar-fill{
            height:100%; border-radius: 999px;
            box-shadow: 0 0 8px currentColor;
        }
        .sg-rul-badge{
            display:inline-block; margin-top: 4px;
            font-family:'JetBrains Mono', monospace; font-size: 0.62rem; font-weight:700;
            padding: 2px 7px; border-radius: 999px; letter-spacing:0.03em;
        }
        .sg-rul-critical{ color:#ff5470; background: rgba(255,84,112,0.1); border:1px solid rgba(255,84,112,0.35); }
        .sg-rul-watch{ color:#ffc857; background: rgba(255,200,87,0.1); border:1px solid rgba(255,200,87,0.35); }
        .sg-rul-stable{ color:#2bffa8; background: rgba(43,255,168,0.1); border:1px solid rgba(43,255,168,0.35); }

        .sg-health-empty{
            text-align:center; padding: 40px 10px; color:#54637a;
            font-family:'JetBrains Mono', monospace; font-size: 0.82rem; letter-spacing:0.04em;
        }
        .sg-health-no-data{
            text-align:center; padding: 18px 4px; color:#54637a;
            font-family:'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing:0.04em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# Helpers
# ==========================================================================
def _classify_status(score: Optional[float]) -> str:
    if score is None:
        return "NO DATA"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "NO DATA"
    if score >= 90:
        return "EXCELLENT"
    if score >= 75:
        return "GOOD"
    if score >= 55:
        return "DEGRADING"
    return "CRITICAL"


def _rul_badge_html(days_to_critical: Optional[float]) -> str:
    if days_to_critical is None:
        return '<span class="sg-rul-badge sg-rul-stable">RUL: STABLE</span>'
    try:
        days = float(days_to_critical)
    except (TypeError, ValueError):
        return '<span class="sg-rul-badge sg-rul-stable">RUL: STABLE</span>'
    if days <= 14:
        cls = "sg-rul-critical"
    elif days <= 45:
        cls = "sg-rul-watch"
    else:
        cls = "sg-rul-stable"
    return f'<span class="sg-rul-badge {cls}">RUL: {days:.1f}d</span>'


def _radial_gauge(score: Optional[float], color: str) -> go.Figure:
    """Build a radial donut-gauge Plotly figure for the overall health score."""
    display_value = score if score is not None else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=display_value,
            number=dict(
                suffix="" if score is not None else "",
                font=dict(family="Orbitron, sans-serif", size=26, color="#f4fbff" if score is not None else "#54637a"),
                valueformat=".1f" if score is not None else "",
            ),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=0, showticklabels=False, visible=False),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(255,255,255,0.05)",
                borderwidth=0,
                shape="angular",
            ),
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )
    fig.update_layout(
        height=150,
        margin=dict(l=8, r=8, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c7d6e6"),
    )
    if score is None:
        fig.data[0].value = 0
        fig.data[0].number.font.color = "#54637a"
    return fig


def _sensor_bar_html(sensor_name: str, sensor_data: Dict[str, Any], color: str) -> str:
    v_score = sensor_data.get("score")
    rul = sensor_data.get("days_to_critical")
    score_str = f"{v_score:.1f}/100" if isinstance(v_score, (int, float)) else "N/A"
    width_pct = max(0.0, min(100.0, float(v_score))) if isinstance(v_score, (int, float)) else 0.0
    display_name = sensor_name.replace("_", " ").title()
    return f"""
    <div class="sg-sensor-row">
        <div class="sg-sensor-label-row">
            <span><b>{display_name}</b></span>
            <span>{score_str}</span>
        </div>
        <div class="sg-sensor-bar-track">
            <div class="sg-sensor-bar-fill" style="width:{width_pct}%; background:{color}; color:{color};"></div>
        </div>
        {_rul_badge_html(rul)}
    </div>
    """


def _render_station_card(col, station_id: str, city: str, summary: Optional[Dict[str, Any]]) -> None:
    with col:
        if summary is None:
            st.markdown(
                f"""
                <div class="sg-health-card" style="--sg-h-color:{_STATUS_COLORS['NO DATA']};">
                    <div class="sg-health-card-head">
                        <div>
                            <div class="sg-health-station">{station_id}</div>
                            <div class="sg-health-city">{city}</div>
                        </div>
                        <span class="sg-health-status-badge">NO DATA</span>
                    </div>
                    <div class="sg-health-no-data">No prognostic health record for this station in the current dataset.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        score = summary.get("overall_score")
        status = str(summary.get("status") or _classify_status(score)).upper()
        if status not in _STATUS_COLORS:
            status = _classify_status(score)
        urgency = summary.get("urgency", "N/A")
        color = _STATUS_COLORS[status]

        st.markdown(
            f"""
            <div class="sg-health-card" style="--sg-h-color:{color}; --sg-h-shadow:{color}55;">
                <div class="sg-health-card-head">
                    <div>
                        <div class="sg-health-station">{station_id}</div>
                        <div class="sg-health-city">{city}</div>
                    </div>
                    <div style="text-align:right;">
                        <span class="sg-health-status-badge">{status}</span>
                        <div class="sg-health-urgency">URGENCY: {urgency}</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        fig = _radial_gauge(score, color)
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False, "staticPlot": True},
            key=f"sg_health_gauge_{station_id}",
        )

        sensors = summary.get("sensors", {}) or {}
        if sensors:
            sensor_html = "".join(_sensor_bar_html(name, data, color) for name, data in sensors.items())
            st.markdown(sensor_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="sg-health-no-data">No per-sensor breakdown reported.</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# Public entrypoint
# ==========================================================================
def render_station_health(demo_data: Dict[str, Any], station_metadata: Dict[str, Any]) -> None:
    """Render the SkyGuard AI Prognostic Health & RUL page.

    Args:
        demo_data: Pipeline / demo results dict. Only
            ``demo_data["station_health_summaries"]`` is read — a list of
            ``{"station_id", "overall_score", "status", "urgency",
            "sensors": {sensor_name: {"score", "days_to_critical"}}}``
            dicts. Missing sub-fields render as "N/A" / "Stable" rather than
            an invented value.
        station_metadata: Real station registry (station_id -> {"city",
            "lat", "lon", "elev"}) used only to resolve display names — it
            is also the source of which stations are listed at all, so a
            station with no health record still appears, explicitly marked
            "NO DATA".

    Each station renders as a responsive glass card containing a radial
    (donut) gauge for the overall prognostic health score, per-sensor score
    bars, and a Remaining Useful Life (RUL) badge per sensor.
    """
    _inject_health_css()
    demo_data = demo_data or {}
    station_metadata = station_metadata or {}

    summaries = demo_data.get("station_health_summaries", []) or []
    summary_by_id = {s.get("station_id"): s for s in summaries if s.get("station_id")}

    st.markdown('<div class="sg-health-title">🩺 Prognostic Health & Remaining Useful Life</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sg-health-caption">HARDWARE DEGRADATION TRACKING · DRIFT ACCUMULATION · RUL PROJECTIONS (0–100 SCORE)</div>',
        unsafe_allow_html=True,
    )

    if not station_metadata:
        st.markdown('<div class="sg-health-empty">NO STATION REGISTRY PROVIDED</div>', unsafe_allow_html=True)
        return

    # Build the station list, joining real metadata with any real health data
    station_ids = sorted(station_metadata.keys())
    statuses_present = set()
    for sid in station_ids:
        s = summary_by_id.get(sid)
        score = s.get("overall_score") if s else None
        status = str(s.get("status") or _classify_status(score)).upper() if s else "NO DATA"
        if status not in _STATUS_COLORS:
            status = _classify_status(score)
        statuses_present.add(status)

    legend_chips = "".join(
        f'<div class="sg-health-chip" style="color:{_STATUS_COLORS[s]};">'
        f'<span class="sg-health-dot" style="background:{_STATUS_COLORS[s]};"></span>{s}</div>'
        for s in _STATUS_ORDER
        if s in statuses_present
    )
    st.markdown(f'<div class="sg-health-legend">{legend_chips}</div>', unsafe_allow_html=True)

    # Responsive card grid: 3 per row (Streamlit columns stack automatically
    # on narrow / mobile viewports).
    cards_per_row = 3
    for i in range(0, len(station_ids), cards_per_row):
        row_ids = station_ids[i : i + cards_per_row]
        cols = st.columns(cards_per_row, gap="medium")
        for col, sid in zip(cols, row_ids):
            city = (station_metadata.get(sid, {}) or {}).get("city", sid)
            _render_station_card(col, sid, city, summary_by_id.get(sid))
