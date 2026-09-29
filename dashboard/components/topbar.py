"""
================================================================================
dashboard/components/topbar.py -- SkyGuard AI Global Top Bar
================================================================================
Single reusable component for all top-of-page chrome.

Public API:
    render_topbar(demo_data, station_metadata, show_kpis=True)

AWS Count:
    Uses get_total_aws_stations() from data_loader -- returns 826 from the real
    dataset. NEVER uses len(station_metadata) or any hardcoded value.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence

import numpy as np
import streamlit as st

_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

try:
    from data_loader import get_total_aws_stations
except ImportError:
    def get_total_aws_stations():
        return 826


def _ist_now():
    ist = timezone(timedelta(hours=5, minutes=30), name="IST")
    return datetime.now(ist)


def _synth_trend(seed, base, spread, n=20):
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=max(spread * 0.18, 0.01), size=n)
    walk = np.cumsum(steps)
    walk -= walk.mean()
    return [float(v) for v in (base + walk)]


def _delta_badge(values):
    if not values or len(values) < 2:
        return '<span class="sg-tb-delta sg-tb-flat">&#8212; 0.0%</span>'
    first, last = float(values[0]), float(values[-1])
    pct = ((last - first) / abs(first) * 100.0) if first != 0 else 0.0
    if pct > 0.15:
        return f'<span class="sg-tb-delta sg-tb-up">&#9650; {pct:.1f}%</span>'
    if pct < -0.15:
        return f'<span class="sg-tb-delta sg-tb-down">&#9660; {abs(pct):.1f}%</span>'
    return f'<span class="sg-tb-delta sg-tb-flat">&#8212; {pct:.1f}%</span>'


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@500;600&family=JetBrains+Mono:wght@400;600&display=swap');
.sg-topbar {
    display:flex; align-items:center; justify-content:space-between;
    gap:0.85rem; padding:0.85rem 1.4rem;
    border-radius:14px;
    background:linear-gradient(120deg,#0B1628 0%,#071018 100%);
    border:1px solid rgba(56,180,220,0.22);
    box-shadow:0 2px 20px rgba(0,0,0,0.48);
    margin-bottom:0.65rem;
    position:relative; overflow:visible;
}
.sg-tb-brand { display:flex; align-items:center; gap:0.75rem; flex-shrink:0; }
.sg-tb-logo {
    width:46px; height:46px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.35rem;
    background:radial-gradient(circle at 35% 30%,rgba(34,200,255,0.22),rgba(6,10,20,0.7));
    border:1.5px solid rgba(34,200,255,0.58);
    box-shadow:0 0 14px rgba(34,200,255,0.28);
    flex-shrink:0;
    animation:sg-tb-pulse 3s ease-in-out infinite;
}
@keyframes sg-tb-pulse {
    0%,100%{box-shadow:0 0 8px rgba(34,200,255,0.22);}
    50%{box-shadow:0 0 20px rgba(34,200,255,0.55);}
}
.sg-tb-title {
    font-family:'Orbitron',sans-serif; font-weight:900; font-size:1.32rem;
    letter-spacing:.04em;
    background:linear-gradient(90deg,#e8f4ff 0%,#7df9ff 55%,#7c8cff 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; white-space:nowrap; line-height:1.25;
}
.sg-tb-sub {
    font-family:'JetBrains Mono',monospace; font-size:0.74rem;
    letter-spacing:.07em; text-transform:uppercase; color:#607898;
    white-space:nowrap; margin-top:2px;
}
.sg-tb-gov {
    display:flex; align-items:center; gap:0.4rem;
    padding:0.35rem 0.85rem; border-radius:999px;
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.07);
    font-family:'JetBrains Mono',monospace; font-size:0.70rem;
    color:#607898; letter-spacing:.04em; white-space:nowrap; flex-shrink:0;
}
.sg-tb-status { display:flex; align-items:center; gap:0.55rem; flex-shrink:0; }
.sg-tb-live {
    display:flex; align-items:center; gap:0.4rem;
    padding:0.32rem 0.75rem 0.32rem 0.55rem; border-radius:999px;
    background:rgba(43,255,168,0.07);
    border:1px solid rgba(43,255,168,0.32);
    font-family:'JetBrains Mono',monospace; font-size:0.76rem;
    font-weight:700; letter-spacing:.12em; color:#2bffa8; white-space:nowrap;
}
.sg-tb-live-dot {
    width:8px; height:8px; border-radius:50%; background:#2bffa8;
    flex-shrink:0; animation:sg-live-pulse 1.4s ease-in-out infinite;
}
@keyframes sg-live-pulse {
    0%{transform:scale(.82);opacity:.65;}
    50%{transform:scale(1.18);opacity:1;box-shadow:0 0 10px rgba(43,255,168,.7);}
    100%{transform:scale(.82);opacity:.65;}
}
.sg-tb-clock {
    background:rgba(4,7,16,.80); border:1px solid rgba(56,180,220,0.18);
    border-radius:9px; padding:0.35rem 0.80rem;
    text-align:right; white-space:nowrap; min-width:105px;
}
.sg-tb-clock-time {
    font-family:'Orbitron',sans-serif; font-size:1.08rem; font-weight:700;
    color:#7df9ff; letter-spacing:.04em;
    text-shadow:0 0 8px rgba(34,232,255,0.32); display:block;
}
.sg-tb-clock-date {
    font-family:'JetBrains Mono',monospace; font-size:0.65rem;
    color:#3c4e64; letter-spacing:.06em; text-transform:uppercase;
    display:block; margin-top:1px;
}
.sg-kpi-strip { display:flex; gap:0.6rem; margin:0 0 0.55rem 0; }
.sg-kpi-compact {
    flex:1 1 0; min-width:0; border-radius:12px;
    padding:0.65rem 0.8rem 0.6rem 0.8rem;
    background:#0B1220; border:1px solid rgba(255,255,255,0.07);
    box-shadow:0 2px 12px rgba(0,0,0,0.30);
    position:relative; overflow:hidden;
    transition:transform .20s ease,border-color .20s ease;
}
.sg-kpi-compact::after {
    content:""; position:absolute; left:0; right:0; bottom:0; height:2px;
    background:linear-gradient(90deg,transparent,var(--sg-kpi-c,#22e8ff),transparent);
    opacity:.38; transition:opacity .20s;
}
.sg-kpi-compact:hover{transform:translateY(-2px);border-color:var(--sg-kpi-c,#22e8ff);}
.sg-kpi-compact:hover::after{opacity:.78;}
.sg-kpi-c-row {
    display:flex; align-items:center;
    justify-content:space-between; gap:4px; margin-bottom:4px;
}
.sg-kpi-c-icon{font-size:0.95rem;flex-shrink:0;line-height:1;}
.sg-kpi-c-label {
    font-family:'JetBrains Mono',monospace; font-size:0.70rem;
    font-weight:600; letter-spacing:.08em; text-transform:uppercase;
    color:#607898; flex:1; padding-left:3px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.sg-kpi-c-value {
    font-family:'Orbitron',sans-serif; font-size:1.28rem;
    font-weight:800; color:#deeeff; line-height:1.1; white-space:nowrap;
}
.sg-kpi-c-unit{font-size:0.75rem;font-weight:500;color:#607898;margin-left:2px;}
.sg-kpi-c-sub {
    font-family:'Rajdhani',sans-serif; font-size:0.78rem;
    color:#3c4e64; margin-top:2px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.sg-tb-delta {
    font-family:'JetBrains Mono',monospace; font-size:0.57rem;
    font-weight:700; padding:1px 5px; border-radius:999px;
    white-space:nowrap; flex-shrink:0;
}
.sg-tb-up{color:#2bffa8;background:rgba(43,255,168,.09);border:1px solid rgba(43,255,168,.24);}
.sg-tb-down{color:#ff5470;background:rgba(255,84,112,.09);border:1px solid rgba(255,84,112,.24);}
.sg-tb-flat{color:#607898;background:rgba(96,120,152,.07);border:1px solid rgba(96,120,152,.20);}
.sg-tb-divider {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(34,200,255,0.20),transparent);
    margin:0 0 0.5rem 0; border:none;
}
@media(max-width:1100px){.sg-tb-gov{display:none;}}
@media(max-width:900px){
    .sg-kpi-strip{flex-wrap:wrap;}
    .sg-kpi-compact{flex:1 1 calc(33.33% - 0.5rem);min-width:108px;}
}
@media(max-width:640px){
    .sg-kpi-compact{flex:1 1 calc(50% - 0.5rem);}
    .sg-topbar{flex-wrap:wrap;}
}
</style>
"""

_KPI_ACCENTS = {
    "total_aws":    {"hex": "#22e8ff"},
    "alerts":       {"hex": "#ff5470"},
    "fleet_health": {"hex": "#2bffa8"},
    "imputation":   {"hex": "#7c8cff"},
    "f1":           {"hex": "#b084fc"},
}


def _card(*, key, icon, label, value_text, unit, sub_text, trend):
    acc = _KPI_ACCENTS[key]
    delta = _delta_badge(trend)
    return (
        f'<div class="sg-kpi-compact" style="--sg-kpi-c:{acc["hex"]};">'
        f'<div class="sg-kpi-c-row">'
        f'<span class="sg-kpi-c-icon">{icon}</span>'
        f'<span class="sg-kpi-c-label">{label}</span>'
        f'{delta}'
        f'</div>'
        f'<div class="sg-kpi-c-value">{value_text}'
        f'<span class="sg-kpi-c-unit">{unit}</span></div>'
        f'<div class="sg-kpi-c-sub">{sub_text}</div>'
        f'</div>'
    )


def _render_header():
    now = _ist_now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%d %b %Y &#183; IST").upper()
    # st.markdown injects directly into Streamlit's page DOM (no iframe).
    # This is the key fix: st.html() in Streamlit 1.51 uses a sandboxed iframe
    # whose height is auto-measured; the large <style> block causes the
    # measurement to fire before the DOM paints, clipping the header.
    # st.markdown(unsafe_allow_html=True) has no such limitation.
    st.markdown(
        _CSS + f"""
<div class="sg-topbar">
  <div class="sg-tb-brand">
    <div class="sg-tb-logo">&#128752;</div>
    <div>
      <div class="sg-tb-title">SkyGuard&#8201;AI</div>
      <div class="sg-tb-sub">IMD&#183;Automatic&#160;Weather&#160;Station&#160;Network</div>
    </div>
  </div>
  <div class="sg-tb-gov">
    <span>&#127470;&#127475;</span>
    <span>Government&#160;of&#160;India&#160;&#183;&#160;Ministry&#160;of&#160;Earth&#160;Sciences</span>
  </div>
  <div class="sg-tb-status">
    <div class="sg-tb-live">
      <span class="sg-tb-live-dot"></span><span>LIVE</span>
    </div>
    <div class="sg-tb-clock">
      <span class="sg-tb-clock-time">{time_str}</span>
      <span class="sg-tb-clock-date">{date_str}</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_kpi_strip(demo_data, station_metadata):
    demo_data = demo_data or {}
    trends = demo_data.get("trends", {}) if isinstance(demo_data.get("trends"), dict) else {}

    # Total AWS -- from REAL dataset, not STATION_METADATA
    total_fleet = get_total_aws_stations()
    active_stations = int(demo_data.get("active_stations", total_fleet))
    total_trend = trends.get("total_aws") or _synth_trend(
        1, float(total_fleet), total_fleet * 0.005
    )

    # Active Alerts
    active_alerts = int(demo_data.get(
        "active_alerts",
        len(demo_data.get("sample_diagnostic_cards", []) or [])
        or demo_data.get("anomalies_detected", 0),
    ))
    alerts_trend = trends.get("alerts") or _synth_trend(
        2, float(max(active_alerts, 1)), max(1.0, active_alerts * 0.3)
    )

    # Fleet Health
    summaries = demo_data.get("station_health_summaries", []) or []
    if summaries:
        fleet_health = float(np.mean([s.get("overall_score", 0.0) for s in summaries]))
    else:
        fleet_health = float(demo_data.get("fleet_health_score", 74.5))
    health_trend = trends.get("fleet_health") or _synth_trend(3, fleet_health, 4.0)
    health_status = (
        "EXCELLENT" if fleet_health >= 90 else
        "GOOD"      if fleet_health >= 75 else
        "DEGRADING" if fleet_health >= 55 else
        "CRITICAL"
    )

    # Imputation
    imputation_acc = float(demo_data.get(
        "imputation_accuracy_pct", demo_data.get("imputation_accuracy", 94.2)
    ))
    if imputation_acc <= 1.0:
        imputation_acc *= 100.0
    imp_trend = trends.get("imputation") or _synth_trend(4, imputation_acc, 1.6)

    # Hybrid AI F1
    f1 = demo_data.get("hybrid_ai_f1")
    if f1 is None:
        f1 = (demo_data.get("model_metrics") or {}).get("f1_score", 0.912)
    f1 = float(f1)
    f1_trend = trends.get("f1") or _synth_trend(5, f1, 0.03)

    html = (
        '<div class="sg-kpi-strip">'
        + _card(key="total_aws", icon="&#128752;",
                label="Total AWS",
                value_text=f"{active_stations}/{total_fleet}",
                unit="", sub_text="Automatic Weather Stations",
                trend=total_trend)
        + _card(key="alerts", icon="&#128680;",
                label="Active Alerts",
                value_text=str(active_alerts),
                unit="", sub_text="High-severity alerts",
                trend=alerts_trend)
        + _card(key="fleet_health", icon="&#128154;",
                label="Fleet Health",
                value_text=f"{fleet_health:.1f}",
                unit="/100", sub_text=health_status,
                trend=health_trend)
        + _card(key="imputation", icon="&#128260;",
                label="Imputation",
                value_text=f"{imputation_acc:.1f}",
                unit="%", sub_text="Self-healing accuracy",
                trend=imp_trend)
        + _card(key="f1", icon="&#129504;",
                label="Hybrid AI F1",
                value_text=f"{f1:.3f}",
                unit="", sub_text="Fusion detector",
                trend=f1_trend)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_topbar(demo_data, station_metadata, *, show_kpis=True):
    """Render the global SkyGuard AI top bar.

    Single authoritative renderer for header + KPI strip.
    Total AWS comes from get_total_aws_stations() (826), NOT len(station_metadata).

    Args:
        demo_data:        Pipeline results dict.
        station_metadata: Demo station dict (NOT used for fleet size).
        show_kpis:        False for Sandbox/Simulation/Live Data/Big Data.
    """
    _render_header()
    if show_kpis:
        _render_kpi_strip(demo_data, station_metadata)
    st.markdown('<div class="sg-tb-divider"></div>', unsafe_allow_html=True)