"""
dashboard/simulation/simulator.py
===================================
Manages simulation state in st.session_state.
Provides get_state() / reset_state() helpers consumed by simulation.py.
"""

from __future__ import annotations
import datetime
import streamlit as st

_KEY = "sim_state"

_DEFAULTS = {
    "station_id":    None,
    "start_date":    datetime.date(2023, 1, 1),
    "end_date":      datetime.date(2025, 12, 31),
    "speed":         "24x",
    "playing":       False,
    "current_idx":   0,
    "fault_active":  False,
    "fault_type":    "Temperature Spike",
    "fault_intensity": 5,
    "fault_duration":  48,
    "fault_result":   None,
    "series_cache_key": None,
}


def get_state() -> dict:
    if _KEY not in st.session_state:
        st.session_state[_KEY] = dict(_DEFAULTS)
    return st.session_state[_KEY]


def reset_state():
    st.session_state[_KEY] = dict(_DEFAULTS)
    # Also clear the cached series
    if "sim_series_cache" in st.session_state:
        del st.session_state["sim_series_cache"]
