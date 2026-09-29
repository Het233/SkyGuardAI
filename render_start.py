#!/usr/bin/env python3
"""
================================================================================
SkyGuard AI — Render Deployment Entrypoint
================================================================================
Render runs:  uvicorn render_start:app --host 0.0.0.0 --port $PORT

This shim simply imports the FastAPI `app` object from api/server.py.
Using this file as the entrypoint means Render does not need the project
root to be the working directory — the sys.path manipulation below ensures
all internal imports (`api.*`, `baseline.*`, `models.*`, etc.) resolve
correctly regardless of where Render mounts the repo.
"""
import os
import sys

# Add project root to sys.path so `from api.server import app` works
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.server import app  # noqa: F401 — re-exported for uvicorn

__all__ = ["app"]
