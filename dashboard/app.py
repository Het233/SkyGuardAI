"""
================================================================================
dashboard/app.py — SkyGuard AI Command Center (entrypoint)
================================================================================
Wires the existing page modules together: injects the shared theme, loads the
cached datasets, renders the sidebar and the global top bar, then routes each
page module. All page logic, CSS and data generation live in the modules this
file imports — nothing is duplicated here.

Top-bar rendering is handled EXCLUSIVELY by:
    components/topbar.py  →  render_topbar(demo_data, STATION_METADATA, show_kpis=...)

Do NOT call render_header() or render_kpi_cards() anywhere else.
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from styles import inject_css
from components.sidebar import render_sidebar
from components.topbar import render_topbar

from pages.network_map import render_network_map
from pages.telemetry import render_live_telemetry
from pages.xai_alerts import render_xai_alerts
from pages.station_health import render_station_health
from pages.sandbox import render_sandbox
from pages.simulation import render_simulation
from pages.live_data import render_live_data
from pages.big_data_analytics import render_big_data_analytics

SAMPLE_CSV = os.path.join(ROOT_DIR, "data", "synthetic", "sample_synthetic_anomalies.csv")
RESULTS_JSON = os.path.join(ROOT_DIR, "artifacts", "pipeline_demo_results.json")

# IMD Automatic Weather Station registry: station_id -> city / lat / lon / elev(m)
STATION_METADATA = {
    "AWS_DEL_001": {"city": "New Delhi", "lat": 28.6139, "lon": 77.2090, "elev": 216},
    "AWS_MUM_002": {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777, "elev": 14},
    "AWS_CHE_003": {"city": "Chennai", "lat": 13.0827, "lon": 80.2707, "elev": 6},
    "AWS_KOL_004": {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639, "elev": 9},
    "AWS_BLR_005": {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "elev": 920},
    "AWS_HYD_006": {"city": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "elev": 542},
    "AWS_AHM_007": {"city": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "elev": 53},
    "AWS_JAI_008": {"city": "Jaipur", "lat": 26.9124, "lon": 75.7873, "elev": 431},
    "AWS_SHL_009": {"city": "Shimla", "lat": 31.1048, "lon": 77.1734, "elev": 2276},
    "AWS_GUW_010": {"city": "Guwahati", "lat": 26.1445, "lon": 91.7362, "elev": 55},
}

# Pages where the KPI strip should be hidden so data-heavy content starts
# immediately after the header.
_NO_KPI_PAGES = {"sandbox", "simulation", "live_data", "big_data"}


@st.cache_data
def load_sample_dataset() -> pd.DataFrame:
    """Load the synthetic anomaly sample; empty frame if the file is absent."""
    if not os.path.exists(SAMPLE_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(SAMPLE_CSV)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data
def load_pipeline_results() -> dict:
    """Load the pipeline demo results; empty dict if the file is absent."""
    if not os.path.exists(RESULTS_JSON):
        return {}
    try:
        with open(RESULTS_JSON, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    demo_data = load_pipeline_results()
    df_sample = load_sample_dataset()

    try:
        inject_css()
        selected_page = render_sidebar()

        # ── Single global top-bar call ───────────────────────────────────────
        # This is the ONLY place render_topbar is called.
        # render_header() and render_kpi_cards() are intentionally NOT imported
        # or called here — topbar.py handles both.
        render_topbar(
            demo_data,
            STATION_METADATA,
            show_kpis=(selected_page not in _NO_KPI_PAGES),
        )
    except Exception as exc:
        import traceback as _tb
        # Use raw markdown so the error is visible even if the CSS theme is
        # obscuring st.error() (error alerts were styled as dark glass panels).
        st.markdown(
            f"""
            <div style="
                background: rgba(255,84,112,0.18); border: 2px solid #ff5470;
                border-radius: 12px; padding: 18px 20px; margin: 12px 0;
                font-family: 'JetBrains Mono', monospace; color: #ffd6de;
            ">
                <div style="font-size:1.1rem; font-weight:700; margin-bottom:8px;"
                    >&#9888;&#65039; SkyGuard AI — shared component error</div>
                <div style="font-size:0.88rem; color:#ff8099;">
                    <b>{type(exc).__name__}</b>: {exc}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Full traceback", expanded=False):
            st.code(_tb.format_exc(), language="python")
        return

    routes = {
        "dashboard":    lambda: render_network_map(demo_data, STATION_METADATA),
        "network":      lambda: render_network_map(demo_data, STATION_METADATA),
        "telemetry":    lambda: render_live_telemetry(),
        "ai_diagnostics": lambda: render_xai_alerts(demo_data),
        "health":       lambda: render_station_health(demo_data, STATION_METADATA),
        "sandbox":      lambda: render_sandbox(STATION_METADATA),
        "simulation":   lambda: render_simulation(),
        "live_data":    lambda: render_live_data(),
        "big_data":     lambda: render_big_data_analytics(),
        "reports":      lambda: st.info("Reports module coming soon."),
    }

    page_renderer = routes.get(selected_page)
    if page_renderer is None:
        st.warning("The selected page is unavailable. Returning to the dashboard.")
        page_renderer = routes["dashboard"]

    try:
        page_renderer()
    except Exception as exc:
        # A failing page MUST NOT clear the shell or prevent navigation.
        st.markdown(
            f"""
            <div style="
                background: rgba(255,84,112,0.14); border: 1px solid rgba(255,84,112,0.6);
                border-radius: 12px; padding: 14px 18px; margin: 8px 0;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.84rem; color: #ffd6de;
            ">
                &#9888; Page render error — select another page from the sidebar.<br>
                <b style="color:#ff8099;">{type(exc).__name__}</b>: {exc}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.exception(exc)

    st.markdown(
        """
        <div style="text-align:center; padding:24px; color:#64748b;">
            SkyGuard AI • Smart India Hackathon 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


main()
