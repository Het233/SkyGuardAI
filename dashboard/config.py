"""
================================================================================
dashboard/config.py — SkyGuard AI Runtime Configuration
================================================================================
Single source of truth for all environment-driven configuration values consumed
by the Streamlit frontend.

Usage in any page or component:
    from config import BACKEND_URL

Priority order for BACKEND_URL:
    1. Streamlit Cloud secrets  (st.secrets["BACKEND_URL"])
    2. OS environment variable  (BACKEND_URL=...)
    3. .env file fallback       (not implemented — use env var or secrets)
    4. Hardcoded localhost default for local dev

This module never crashes — every lookup is wrapped in a try/except so a
missing secret or env var always falls back to the local default.
================================================================================
"""
from __future__ import annotations

import os

import streamlit as st


def _get_backend_url() -> str:
    """Return the configured backend base URL.

    Tries st.secrets first (Streamlit Cloud), then the OS env var, then falls
    back to http://localhost:8000 for local development.
    """
    # 1. Streamlit Cloud / secrets.toml
    try:
        url = st.secrets.get("BACKEND_URL", None)
        if url:
            return url.rstrip("/")
    except Exception:
        pass

    # 2. OS environment variable
    url = os.getenv("BACKEND_URL", "")
    if url:
        return url.rstrip("/")

    # 3. Local development default
    return "http://localhost:8000"


# ---------------------------------------------------------------------------
# Public constants — import these directly in page/component modules
# ---------------------------------------------------------------------------

BACKEND_URL: str = _get_backend_url()
"""Base URL of the SkyGuard AI FastAPI backend.

Examples:
    Local:  http://localhost:8000
    Render: https://skyguard-api.onrender.com
"""

# Convenience endpoint constants
HEALTH_ENDPOINT     = f"{BACKEND_URL}/health"
INGEST_SINGLE       = f"{BACKEND_URL}/api/v1/ingest/single"
INGEST_BATCH        = f"{BACKEND_URL}/api/v1/ingest/batch"
ALL_HEALTH_ENDPOINT = f"{BACKEND_URL}/api/v1/stations/health"
ALERTS_ENDPOINT     = f"{BACKEND_URL}/api/v1/alerts/active"
