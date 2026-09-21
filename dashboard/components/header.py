"""
================================================================================
dashboard/components/header.py — SkyGuard AI Command Center Header
================================================================================
Renders the top mission-control header bar: brand mark + title/subtitle,
Government of India / IMD branding badge, a current IST clock, and an
animated "LIVE" telemetry badge.

Requires `inject_css()` (from dashboard.styles) to have been called earlier
in the script so the shared `.sg-*` classes resolve for the static portion
of the header rendered via `st.markdown`.
"""

from datetime import datetime, timedelta, timezone

import streamlit as st


def _render_brand_block() -> None:
    """Render the left-hand brand block: logo, title, subtitle, gov badge."""
    st.markdown(
        """
        <div class="sg-header-wrap" style="margin-bottom:0;">
            <div class="sg-brand-block">
                <div class="sg-logo-ring">🛰️</div>
                <div class="sg-title-group">
                    <div class="sg-title">SkyGuard&nbsp;AI</div>
                    <div class="sg-subtitle">
                        <strong>IMD</strong> · India Meteorological Department —
                        Automatic Weather Station Network Surveillance &amp; Command Center
                    </div>
                </div>
                <div class="sg-gov-badge">
                    <span class="sg-flag">🇮🇳</span>
                    <span>GOVERNMENT&nbsp;OF&nbsp;INDIA&nbsp;·&nbsp;MINISTRY&nbsp;OF&nbsp;EARTH&nbsp;SCIENCES</span>
                </div>
            </div>
            <div id="sg-status-anchor" style="min-width: 340px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_live_status() -> None:
    """Render the LIVE badge and clock without mounting an iframe component.

    The former iframe-based HTML component was the source of the client-side
    rerender that cleared the page. This status display is static for the
    current Streamlit run; CSS retains the existing LIVE animation.
    """
    ist = timezone(timedelta(hours=5, minutes=30), name="IST")
    now = datetime.now(ist)
    time_text = now.strftime("%H:%M:%S")
    date_text = now.strftime("%a, %d %b %Y · IST")

    st.html(
        """
        <style>
            .wrap{
                display:flex; align-items:center; justify-content:flex-end;
                gap: 0.9rem; flex-wrap: wrap;
                font-family: 'JetBrains Mono', monospace;
                padding: 2px 2px;
            }
            .live-badge{
                display:flex; align-items:center; gap: 0.45rem;
                padding: 0.45rem 0.85rem 0.45rem 0.65rem;
                border-radius: 999px;
                background: rgba(43, 255, 168, 0.08);
                border: 1px solid rgba(43, 255, 168, 0.45);
                box-shadow: 0 0 16px rgba(43, 255, 168, 0.2);
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                color: #2bffa8;
                white-space: nowrap;
            }
            .live-dot{
                width: 9px; height: 9px;
                border-radius: 50%;
                background: #2bffa8;
                box-shadow: 0 0 8px #2bffa8, 0 0 16px #2bffa8;
                animation: pulse 1.4s ease-in-out infinite;
            }
            @keyframes pulse{
                0%{ transform: scale(0.85); opacity: 0.7; box-shadow: 0 0 4px #2bffa8; }
                50%{ transform: scale(1.15); opacity: 1; box-shadow: 0 0 14px #2bffa8, 0 0 26px #2bffa8; }
                100%{ transform: scale(0.85); opacity: 0.7; box-shadow: 0 0 4px #2bffa8; }
            }
            .clock{
                font-family: 'JetBrains Mono', monospace;
                background: rgba(6,10,20,0.7);
                border: 1px solid rgba(56, 227, 255, 0.18);
                border-radius: 10px;
                padding: 0.5rem 0.9rem;
                text-align: right;
                min-width: 168px;
            }
            .clock-time{
                font-size: 1.05rem;
                font-weight: 700;
                color: #7df9ff;
                letter-spacing: 0.05em;
                text-shadow: 0 0 12px rgba(34,232,255,0.4);
                font-family: 'Orbitron', sans-serif;
            }
            .clock-date{
                font-size: 0.66rem;
                color: #54637a;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-top: 2px;
            }
        </style>
            <div class="wrap">
                <div class="live-badge">
                    <span class="live-dot"></span>
                    <span>LIVE</span>
                </div>
                <div class="clock">
                    <div class="clock-time">__TIME__</div>
                    <div class="clock-date">__DATE__</div>
                </div>
            </div>
        """.replace("__TIME__", time_text).replace("__DATE__", date_text),
    )


def render_header() -> None:
    """Render the full SkyGuard AI / IMD command center header bar.

    Includes:
        - SkyGuard AI title with satellite brand mark
        - IMD (India Meteorological Department) subtitle
        - Government of India branding badge
        - A current IST clock
        - An animated pulsing "LIVE" badge indicating active telemetry

    Call this once per page render, after `dashboard.styles.inject_css()`.
    """
    header_col, clock_col = st.columns([2.6, 1.3], gap="small")
    with header_col:
        _render_brand_block()
    with clock_col:
        st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)
        _render_live_status()

    st.markdown('<div class="sg-divider"></div>', unsafe_allow_html=True)
