"""
================================================================================
dashboard/components/sidebar.py — SkyGuard AI Command Center Sidebar
================================================================================
Renders the left navigation rail: brand mark, primary navigation (Dashboard,
Network, Telemetry, AI Diagnostics, Health, Sandbox, Reports), and a compact
system-status footer.

Requires `inject_css()` (from dashboard.styles) to have been called earlier
in the script so the `.sg-sidebar-*` / nav radio classes resolve.
"""

import os
import sys
from typing import Tuple

import streamlit as st

# Resolve data_loader from dashboard root
_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

try:
    from data_loader import get_total_aws_stations as _get_aws_count
except ImportError:
    def _get_aws_count():
        return 826

# Ordered nav definition: (page key, display label, icon)
NAV_ITEMS: Tuple[Tuple[str, str, str], ...] = (
    ("dashboard", "Dashboard", "🛰️"),
    ("network", "Network", "📡"),
    ("telemetry", "Telemetry", "📈"),
    ("ai_diagnostics", "AI Diagnostics", "🧠"),
    ("health", "Health", "❤️‍🩹"),
    ("sandbox",    "Sandbox",        "⚡"),
    ("simulation", "Simulation Lab", "🔬"),
    ("live_data",  "Live Data",      "📡"),
    ("big_data",   "Big Data",       "📊"),
    ("reports",    "Reports",        "📋"),
)

_SESSION_KEY = "sg_active_page"


def render_sidebar() -> str:
    """Render the SkyGuard AI command-center sidebar navigation.

    Displays the brand mark, the primary navigation list (Dashboard, Network,
    Telemetry, AI Diagnostics, Health, Sandbox, Reports) as a glass/neon
    vertical nav, and a small system-status footer.

    Returns:
        str: the key of the currently selected page (one of "dashboard",
        "network", "telemetry", "ai_diagnostics", "health", "sandbox",
        "reports"), also persisted in ``st.session_state["sg_active_page"]``
        so other components can read it without re-rendering the sidebar.
    """
    labels = [f"{icon}  {label}" for _, label, icon in NAV_ITEMS]
    keys = [key for key, _, _ in NAV_ITEMS]

    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = keys[0]

    with st.sidebar:
        st.markdown(
            """
            <div class="sg-sidebar-brand">
                <div class="sg-sidebar-logo">🛰️</div>
                <div class="sg-sidebar-name">SkyGuard&nbsp;AI</div>
                <div class="sg-sidebar-tag">IMD Command Center</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sg-nav-caption">Navigation</div>', unsafe_allow_html=True)

        current_key = st.session_state[_SESSION_KEY]
        current_index = keys.index(current_key) if current_key in keys else 0

        selected_label = st.radio(
            label="Navigation",
            options=labels,
            index=current_index,
            key="sg_nav_radio",
            label_visibility="collapsed",
        )

        try:
            selected_index = labels.index(selected_label) if selected_label in labels else 0
        except (ValueError, TypeError):
            selected_index = 0
        selected_key = keys[selected_index]
        st.session_state[_SESSION_KEY] = selected_key

        try:
            aws_count = _get_aws_count()
        except Exception:
            aws_count = 826
        st.markdown(
            f"""
            <div class="sg-sidebar-footer">
                <div><span class="sg-dot-ok"></span>ALL SYSTEMS NOMINAL</div>
                <div>{aws_count} AWS STATIONS IN NETWORK</div>
                <div>SkyGuard&nbsp;AI&nbsp;v2.4&nbsp;·&nbsp;IMD&nbsp;GOI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_key