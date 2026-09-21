"""
================================================================================
dashboard/pages/xai_alerts.py — SkyGuard AI XAI Diagnostic Alert Center
================================================================================
Renders premium, expandable Explainable-AI (XAI) diagnostic alert cards:
colored severity borders, a physical-evidence checklist, a prescriptive SOP
/ maintenance panel, and (when available) the full formatted report from
`DiagnosticCardGenerator.format_markdown()`.

Data policy: reads only from `demo_data["sample_diagnostic_cards"]`. Every
field is pulled defensively with `.get(...)`; missing fields render as
"N/A" or an empty-state line rather than an invented value, and a station
with no alerts simply produces no card.

Requires `dashboard.styles.inject_css()` to have been called earlier in the
page so shared tokens are available; this module injects a small amount of
additional, page-specific CSS on top of that shared theme.
"""

import html
from typing import Any, Dict, List, Optional

import streamlit as st

_SEVERITY_STYLE: Dict[str, Dict[str, str]] = {
    "CRITICAL": {"color": "#ff5470", "icon": "🔴"},
    "HIGH": {"color": "#ff5470", "icon": "🔴"},
    "ANOMALY": {"color": "#ffc857", "icon": "🟠"},
    "MODERATE": {"color": "#ffc857", "icon": "🟠"},
    "SUSPICIOUS": {"color": "#22e8ff", "icon": "🔵"},
    "LOW": {"color": "#2bffa8", "icon": "🟢"},
}
_DEFAULT_SEVERITY_STYLE = {"color": "#8aa2bd", "icon": "⚪"}
_SEVERITY_PRIORITY = ["CRITICAL", "HIGH", "ANOMALY", "MODERATE", "SUSPICIOUS", "LOW"]


# ==========================================================================
# CSS
# ==========================================================================
def _inject_xai_css() -> None:
    st.markdown(
        """
        <style>
        .sg-xai-title{
            font-family:'Orbitron', sans-serif; font-size: 1.3rem; font-weight:700;
            color:#e7f6ff; letter-spacing:0.03em;
        }
        .sg-xai-caption{
            font-family:'JetBrains Mono', monospace; font-size: 0.72rem;
            color:#8aa2bd; letter-spacing:0.04em; margin-bottom: 10px;
        }
        .sg-xai-summary-row{ display:flex; flex-wrap:wrap; gap:10px; margin: 6px 0 16px 0; }
        .sg-xai-count-chip{
            display:flex; align-items:center; gap:6px;
            font-family:'JetBrains Mono', monospace; font-size: 0.7rem;
            letter-spacing:0.03em; color:#c7d6e6;
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 999px; padding: 5px 12px;
        }
        .sg-xai-count-dot{ width:9px; height:9px; border-radius:50%; box-shadow: 0 0 8px currentColor; }

        .sg-xai-card{
            display:block;
            border-radius: 16px;
            margin-bottom: 12px;
            background: linear-gradient(160deg, rgba(16,24,42,0.62), rgba(8,13,24,0.52));
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 4px solid var(--sg-xai-color, #22e8ff);
            backdrop-filter: blur(18px) saturate(150%);
            -webkit-backdrop-filter: blur(18px) saturate(150%);
            box-shadow: 0 8px 26px rgba(0,0,0,0.35);
            overflow: hidden;
            transition: box-shadow 0.25s ease, border-color 0.25s ease, transform 0.2s ease;
        }
        .sg-xai-card:hover{
            box-shadow: 0 12px 34px rgba(0,0,0,0.45), 0 0 26px 0 var(--sg-xai-shadow, rgba(34,232,255,0.22));
            transform: translateY(-2px);
        }
        .sg-xai-card > summary{
            list-style: none;
            cursor: pointer;
            padding: 14px 18px;
            display:flex; align-items:center; justify-content:space-between;
            gap: 12px; flex-wrap: wrap;
        }
        .sg-xai-card > summary::-webkit-details-marker{ display:none; }
        .sg-xai-summary-left{ display:flex; align-items:center; gap:10px; flex-wrap: wrap; min-width: 0; }
        .sg-xai-chevron{
            display:inline-block; color: var(--sg-xai-color, #22e8ff);
            font-size: 0.85rem; transition: transform 0.22s ease; flex-shrink:0;
        }
        .sg-xai-card[open] > summary .sg-xai-chevron{ transform: rotate(90deg); }
        .sg-xai-station{
            font-family:'JetBrains Mono', monospace; font-weight:700; font-size: 0.86rem;
            color:#e7f6ff; letter-spacing:0.02em; white-space: nowrap;
        }
        .sg-xai-fault{
            font-family:'Rajdhani', sans-serif; font-weight:600; font-size: 0.95rem;
            color:#dbe9f7; min-width:0;
        }
        .sg-xai-summary-right{ display:flex; align-items:center; gap:8px; flex-wrap: wrap; }
        .sg-xai-badge{
            font-family:'JetBrains Mono', monospace; font-size: 0.66rem; font-weight:700;
            letter-spacing:0.06em; padding: 3px 9px; border-radius:999px; white-space: nowrap;
        }
        .sg-xai-badge-severity{
            color: var(--sg-xai-color, #22e8ff);
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--sg-xai-color, #22e8ff);
        }
        .sg-xai-badge-conf{
            color:#c7d6e6; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12);
        }
        .sg-xai-time{
            font-family:'JetBrains Mono', monospace; font-size: 0.68rem; color:#54637a;
        }

        .sg-xai-body{
            padding: 4px 18px 18px 18px;
            border-top: 1px solid rgba(255,255,255,0.06);
        }
        .sg-xai-meta-row{
            display:flex; flex-wrap:wrap; gap: 16px; margin: 12px 0 14px 0;
            font-family:'JetBrains Mono', monospace; font-size: 0.72rem; color:#8aa2bd;
        }
        .sg-xai-meta-row b{ color:#c7d6e6; }

        .sg-xai-grid{
            display:grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        @media (max-width: 900px){
            .sg-xai-grid{ grid-template-columns: 1fr; }
        }
        .sg-xai-panel{
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 12px 14px;
        }
        .sg-xai-panel h5{
            margin: 0 0 8px 0; font-family:'Orbitron', sans-serif; font-size: 0.78rem;
            color:#e7f6ff; letter-spacing:0.03em;
        }
        .sg-xai-panel ul{ margin:0; padding-left: 18px; }
        .sg-xai-panel li{
            font-family:'Rajdhani', sans-serif; font-size: 0.86rem; color:#c7d6e6;
            margin-bottom: 5px; line-height: 1.35;
        }
        .sg-xai-empty-line{
            font-family:'JetBrains Mono', monospace; font-size: 0.72rem; color:#54637a;
        }
        .sg-xai-sop-row{ margin-bottom: 8px; font-family:'Rajdhani', sans-serif; font-size: 0.86rem; color:#c7d6e6; }
        .sg-xai-sop-row b{ color:#7df9ff; font-family:'JetBrains Mono', monospace; font-size:0.78rem; letter-spacing:0.03em; }
        .sg-xai-sop-task{
            margin-top: 8px; padding: 8px 10px; border-radius: 8px;
            background: rgba(34,232,255,0.06); border: 1px solid rgba(34,232,255,0.2);
            font-size: 0.84rem; color:#dbe9f7;
        }

        .sg-xai-report{
            margin-top: 14px; border-radius: 10px; border: 1px dashed rgba(255,255,255,0.14);
            padding: 10px 12px; background: rgba(0,0,0,0.18);
        }
        .sg-xai-report summary{
            cursor: pointer; font-family:'JetBrains Mono', monospace; font-size: 0.7rem;
            color:#8aa2bd; letter-spacing:0.05em; list-style:none;
        }
        .sg-xai-report summary::-webkit-details-marker{ display:none; }
        .sg-xai-report pre{
            white-space: pre-wrap; word-break: break-word;
            font-family:'JetBrains Mono', monospace; font-size: 0.78rem;
            color:#c7d6e6; margin-top: 10px; line-height:1.5;
        }
        .sg-xai-empty-state{
            text-align:center; padding: 40px 10px; color:#54637a;
            font-family:'JetBrains Mono', monospace; font-size: 0.82rem; letter-spacing:0.04em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# DiagnosticCardGenerator — best-effort import, never fatal
# ==========================================================================
def _get_diagnostic_card_generator():
    for module_path in (
        "explainability.diagnostic_cards",
        "explainability.cards",
        "api.schemas",
        "api.diagnostics",
    ):
        try:
            module = __import__(module_path, fromlist=["DiagnosticCardGenerator"])
            generator = getattr(module, "DiagnosticCardGenerator", None)
            if generator is not None:
                return generator
        except ImportError:
            continue
    return None


def _format_full_report(generator, card: Dict[str, Any]) -> Optional[str]:
    """Best-effort call into `DiagnosticCardGenerator.format_markdown`.

    Returns None (no fabricated report) if the generator is unavailable or
    raises on this card shape — the custom evidence/SOP panels still render
    either way.
    """
    if generator is None:
        return None
    try:
        return generator.format_markdown(card)
    except Exception:
        return None


# ==========================================================================
# Rendering
# ==========================================================================
def _severity_style(severity: str) -> Dict[str, str]:
    return _SEVERITY_STYLE.get(severity.upper(), _DEFAULT_SEVERITY_STYLE)


def _sort_key(card: Dict[str, Any]):
    hdr = card.get("alert_header", {}) or {}
    severity = str(hdr.get("severity", "")).upper()
    prio = _SEVERITY_PRIORITY.index(severity) if severity in _SEVERITY_PRIORITY else len(_SEVERITY_PRIORITY)
    return (prio, str(hdr.get("timestamp", "")))


def _render_card_html(card: Dict[str, Any], generator, open_first: bool) -> str:
    hdr = card.get("alert_header", {}) or {}
    attrib = card.get("sensor_attribution", {}) or {}
    evid = card.get("evidence_breakdown", {}) or {}
    maint = card.get("maintenance_prescription", {}) or {}

    station_id = hdr.get("station_id", "UNKNOWN")
    severity = str(hdr.get("severity", "N/A")).upper()
    urgency = hdr.get("urgency", "N/A")
    fault_title = hdr.get("fault_title") or hdr.get("predicted_fault") or "Unclassified Fault"
    timestamp = hdr.get("timestamp", "—")
    confidence = hdr.get("confidence")
    confidence_str = f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else "N/A"
    style = _severity_style(severity)

    # Physical evidence checklist
    evidence_items: List[str] = list(evid.get("summary_points", []) or [])
    spatial = evid.get("spatial_disparity", {}) or {}
    if spatial.get("has_disparity"):
        evidence_items.append(f"Spatial Disparity: {spatial.get('description', 'N/A')}")
    thermo = evid.get("thermodynamic_state", {}) or {}
    if thermo.get("has_violation"):
        evidence_items.append(f"Thermodynamic Violation: {thermo.get('description', 'N/A')}")

    if evidence_items:
        evidence_html = "<ul>" + "".join(f"<li>{html.escape(str(pt))}</li>" for pt in evidence_items) + "</ul>"
    else:
        evidence_html = '<div class="sg-xai-empty-line">No physical evidence points reported.</div>'

    protocol = maint.get("protocol")
    action = maint.get("action")
    technician_task = maint.get("technician_task")
    sop_rows = ""
    if protocol:
        sop_rows += f'<div class="sg-xai-sop-row"><b>PROTOCOL</b><br>{html.escape(str(protocol))}</div>'
    if action:
        sop_rows += f'<div class="sg-xai-sop-row"><b>PRIMARY ACTION</b><br>{html.escape(str(action))}</div>'
    if not sop_rows:
        sop_rows = '<div class="sg-xai-empty-line">No maintenance prescription reported.</div>'
    task_html = f'<div class="sg-xai-sop-task">🔧 {html.escape(str(technician_task))}</div>' if technician_task else ""

    primary_culprit = attrib.get("primary_culprit", "N/A")

    report_text = _format_full_report(generator, card)
    report_html = ""
    if report_text:
        report_html = f"""
        <details class="sg-xai-report">
            <summary>▾ FULL DIAGNOSTIC REPORT (DiagnosticCardGenerator)</summary>
            <pre>{html.escape(report_text)}</pre>
        </details>
        """

    open_attr = " open" if open_first else ""

    return f"""
    <details class="sg-xai-card" style="--sg-xai-color:{style['color']}; --sg-xai-shadow:{style['color']}55;"{open_attr}>
        <summary>
            <div class="sg-xai-summary-left">
                <span class="sg-xai-chevron">▸</span>
                <span>{style['icon']}</span>
                <span class="sg-xai-station">{html.escape(str(station_id))}</span>
                <span class="sg-xai-fault">{html.escape(str(fault_title))}</span>
            </div>
            <div class="sg-xai-summary-right">
                <span class="sg-xai-badge sg-xai-badge-severity">{severity}</span>
                <span class="sg-xai-badge sg-xai-badge-conf">CONF {confidence_str}</span>
                <span class="sg-xai-time">{html.escape(str(timestamp))}</span>
            </div>
        </summary>
        <div class="sg-xai-body">
            <div class="sg-xai-meta-row">
                <span><b>Primary Culprit:</b> {html.escape(str(primary_culprit))}</span>
                <span><b>Urgency:</b> {html.escape(str(urgency))}</span>
            </div>
            <div class="sg-xai-grid">
                <div class="sg-xai-panel">
                    <h5>📋 Physical Evidence</h5>
                    {evidence_html}
                </div>
                <div class="sg-xai-panel">
                    <h5>🛠️ Prescriptive SOP</h5>
                    {sop_rows}
                    {task_html}
                </div>
            </div>
            {report_html}
        </div>
    </details>
    """


def render_xai_alerts(demo_data: Dict[str, Any]) -> None:
    """Render the SkyGuard AI XAI Diagnostic Alert Center page.

    Args:
        demo_data: Pipeline / demo results dict. Only
            ``demo_data["sample_diagnostic_cards"]`` is read — a list of
            alert dicts each shaped as
            ``{"alert_header": {...}, "sensor_attribution": {...},
            "evidence_breakdown": {...}, "maintenance_prescription": {...}}``.
            Missing sub-fields render as explicit empty-state text, never a
            fabricated value.

    Each alert renders as a premium, natively expandable (``<details>``)
    glass card with a severity-colored left border, a confidence badge, a
    physical-evidence checklist, a prescriptive SOP / maintenance panel, and
    — when ``DiagnosticCardGenerator`` is importable and accepts this card
    shape — the full formatted report from
    ``DiagnosticCardGenerator.format_markdown()`` in a collapsible
    sub-section.
    """
    _inject_xai_css()
    demo_data = demo_data or {}
    cards: List[Dict[str, Any]] = demo_data.get("sample_diagnostic_cards", []) or []

    st.markdown('<div class="sg-xai-title">🚨 XAI Diagnostic Alert Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sg-xai-caption">EXPLAINABLE AI · ROOT-CAUSE ATTRIBUTION · PRESCRIPTIVE MAINTENANCE SOP</div>',
        unsafe_allow_html=True,
    )

    if not cards:
        st.markdown('<div class="sg-xai-empty-state">✅ NO ACTIVE HIGH-SEVERITY ALERTS IN CURRENT BUFFER</div>', unsafe_allow_html=True)
        return

    cards_sorted = sorted(cards, key=_sort_key)

    # Severity count chips
    counts: Dict[str, int] = {}
    for c in cards_sorted:
        sev = str((c.get("alert_header", {}) or {}).get("severity", "N/A")).upper()
        counts[sev] = counts.get(sev, 0) + 1
    chips = "".join(
        f'<div class="sg-xai-count-chip" style="color:{_severity_style(sev)["color"]};">'
        f'<span class="sg-xai-count-dot" style="background:{_severity_style(sev)["color"]};"></span>'
        f"{sev} · {n}</div>"
        for sev, n in counts.items()
    )
    st.markdown(f'<div class="sg-xai-summary-row">{chips}</div>', unsafe_allow_html=True)

    generator = _get_diagnostic_card_generator()

    html_blocks = [
        _render_card_html(card, generator, open_first=(idx == 0)) for idx, card in enumerate(cards_sorted)
    ]
    st.markdown("".join(html_blocks), unsafe_allow_html=True)