"""
================================================================================
dashboard/styles.py — SkyGuard AI Command Center Theme Engine
================================================================================
Injects the global glassmorphism / neon-cyan / dark "mission control" theme
used across the SkyGuard AI (IMD) Streamlit dashboard. Import and call
`inject_css()` once, near the top of the app, before rendering other
components.
"""

import streamlit as st


def inject_css() -> None:
    """Inject the SpaceX × IMD Command Center CSS theme into the current page.

    Covers: base dark theme, glassmorphism surfaces, neon-cyan glow accents,
    animated status indicators, custom scrollbars, responsive breakpoints,
    and shared utility classes consumed by header.py / sidebar.py / app.py.
    """
    st.markdown(
        """
<style>
/* ============================================================================
   FONTS
============================================================================ */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ============================================================================
   ROOT TOKENS
============================================================================ */
:root{
    --sg-bg-0:#03050a;
    --sg-bg-1:#060a14;
    --sg-bg-2:#0a1120;
    --sg-panel:rgba(14, 22, 40, 0.55);
    --sg-panel-border:rgba(56, 227, 255, 0.18);
    --sg-panel-border-strong:rgba(56, 227, 255, 0.45);
    --sg-cyan:#22e8ff;
    --sg-cyan-soft:#7df9ff;
    --sg-cyan-dim:rgba(34, 232, 255, 0.35);
    --sg-indigo:#7c8cff;
    --sg-violet:#b084fc;
    --sg-green:#2bffa8;
    --sg-amber:#ffc857;
    --sg-red:#ff5470;
    --sg-text:#e7f6ff;
    --sg-text-dim:#8aa2bd;
    --sg-text-faint:#54637a;
    --sg-radius:16px;
    --sg-radius-sm:10px;
    --sg-glow:0 0 18px rgba(34, 232, 255, 0.35), 0 0 42px rgba(34, 232, 255, 0.12);
    --sg-glow-strong:0 0 24px rgba(34, 232, 255, 0.55), 0 0 70px rgba(34, 232, 255, 0.22);
}

/* ============================================================================
   BASE APP SHELL
============================================================================ */
html, body, [class*="css"]{
    font-family: 'Rajdhani', 'Segoe UI', sans-serif !important;
}

/* DO NOT set position on .stApp — Streamlit needs position:fixed (its own CSS)
   so the container stretches to fill the viewport via right/bottom:0.
   Overriding to 'relative' collapses height to 0px + overflow:hidden = blank page. */
.stApp{
    background:
        radial-gradient(ellipse 90% 60% at 15% -10%, rgba(56, 130, 246, 0.16) 0%, transparent 55%),
        radial-gradient(ellipse 80% 55% at 105% 10%, rgba(176, 132, 252, 0.14) 0%, transparent 55%),
        radial-gradient(ellipse 70% 50% at 50% 110%, rgba(34, 232, 255, 0.10) 0%, transparent 60%),
        linear-gradient(180deg, var(--sg-bg-0) 0%, var(--sg-bg-1) 45%, var(--sg-bg-0) 100%);
    background-attachment: fixed;
    color: var(--sg-text);
}

/* Faint scanline / grid overlay for "mission control" feel */
.stApp::before{
    content:"";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(34, 232, 255, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34, 232, 255, 0.035) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(ellipse 80% 70% at 50% 20%, black 0%, transparent 75%);
}

/* Streamlit 1.51 main content container — covers both old and new class names */
section.main > div.block-container,
section[data-testid="stMain"] > div.block-container,
section[data-testid="stMain"] > div.stMainBlockContainer,
div[data-testid="stMainBlockContainer"]{
    padding-top: 1.1rem;
    padding-bottom: 3rem;
    max-width: 1440px;
    position: relative;
    z-index: 1;
}

#MainMenu, footer, header[data-testid="stHeader"]{
    background: transparent;
}
header[data-testid="stHeader"]{
    background: linear-gradient(180deg, rgba(3,5,10,0.85) 0%, transparent 100%);
    backdrop-filter: blur(6px);
}

/* ============================================================================
   SCROLLBAR
============================================================================ */
::-webkit-scrollbar{ width: 9px; height: 9px; }
::-webkit-scrollbar-track{ background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb{
    background: linear-gradient(180deg, var(--sg-cyan), var(--sg-indigo));
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover{ background: var(--sg-cyan-soft); }

/* ============================================================================
   GLASSMORPHIC PANEL / CARD PRIMITIVES  (shared utility classes)
============================================================================ */
.sg-glass{
    background: var(--sg-panel);
    border: 1px solid var(--sg-panel-border);
    border-radius: var(--sg-radius);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
}
.sg-glass:hover{
    border-color: var(--sg-panel-border-strong);
    box-shadow: var(--sg-glow), 0 8px 32px rgba(0,0,0,0.4);
}

.sg-divider{
    height: 1px;
    width: 100%;
    background: linear-gradient(90deg, transparent, var(--sg-cyan-dim), transparent);
    margin: 0.6rem 0;
    border: none;
}

/* ============================================================================
   STREAMLIT NATIVE WIDGET RE-SKIN
============================================================================ */
div[data-testid="stMetric"]{
    background: var(--sg-panel);
    border: 1px solid var(--sg-panel-border);
    border-radius: var(--sg-radius-sm);
    padding: 14px 16px 10px 16px;
    backdrop-filter: blur(14px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    transition: all 0.25s ease;
}
div[data-testid="stMetric"]:hover{
    border-color: var(--sg-panel-border-strong);
    box-shadow: var(--sg-glow);
    transform: translateY(-2px);
}
div[data-testid="stMetricLabel"]{
    color: var(--sg-text-dim) !important;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem !important;
}
div[data-testid="stMetricValue"]{
    color: var(--sg-cyan-soft) !important;
    text-shadow: 0 0 16px rgba(34,232,255,0.4);
    font-family: 'Orbitron', sans-serif;
}

div[data-testid="stExpander"]{
    background: var(--sg-panel);
    border: 1px solid var(--sg-panel-border);
    border-radius: var(--sg-radius-sm);
    backdrop-filter: blur(14px);
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"]{
    gap: 4px;
    background: rgba(10,17,32,0.5);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--sg-panel-border);
}
.stTabs [data-baseweb="tab"]{
    color: var(--sg-text-dim);
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"]{
    background: linear-gradient(180deg, rgba(34,232,255,0.16), rgba(34,232,255,0.04)) !important;
    color: var(--sg-cyan-soft) !important;
    box-shadow: inset 0 0 0 1px var(--sg-cyan-dim), 0 0 14px rgba(34,232,255,0.18);
}

.stButton > button{
    background: linear-gradient(135deg, rgba(34,232,255,0.14), rgba(124,140,255,0.10));
    border: 1px solid var(--sg-panel-border-strong);
    color: var(--sg-cyan-soft);
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
}
.stButton > button:hover{
    box-shadow: var(--sg-glow-strong);
    border-color: var(--sg-cyan);
    color: #fff;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"]{
    background: linear-gradient(135deg, var(--sg-cyan), var(--sg-indigo));
    color: #04121a;
    border: none;
    box-shadow: var(--sg-glow);
}

div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input{
    background: rgba(10,17,32,0.65) !important;
    border: 1px solid var(--sg-panel-border) !important;
    color: var(--sg-text) !important;
    border-radius: 8px !important;
}

.stProgress > div > div{
    background: linear-gradient(90deg, var(--sg-cyan), var(--sg-indigo)) !important;
    box-shadow: 0 0 10px rgba(34,232,255,0.5);
}

/* Alert overrides — keep glassmorphism for info/success, but make errors and
   warnings visible so they are never mistaken for a blank page. */
div[data-testid="stAlert"]{
    border-radius: var(--sg-radius-sm);
    backdrop-filter: blur(12px);
}
/* Info / success use the dark glass panel */
div[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
div[data-testid="stAlert"][data-baseweb="notification"][kind="success"]{
    background: var(--sg-panel);
    border: 1px solid var(--sg-panel-border);
}
/* Error and warning must remain visually distinct against the dark background */
div[data-testid="stAlert"][data-baseweb="notification"][kind="error"]{
    background: rgba(255, 84, 112, 0.14) !important;
    border: 1px solid rgba(255, 84, 112, 0.60) !important;
    color: #ffd6de !important;
}
div[data-testid="stAlert"][data-baseweb="notification"][kind="warning"]{
    background: rgba(255, 200, 87, 0.12) !important;
    border: 1px solid rgba(255, 200, 87, 0.55) !important;
    color: #fff3cc !important;
}

/* ============================================================================
   SIDEBAR
============================================================================ */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(4,7,14,0.96) 0%, rgba(6,10,20,0.98) 100%);
    border-right: 1px solid var(--sg-panel-border);
}
section[data-testid="stSidebar"] > div{
    padding-top: 0.6rem;
}

/* ============================================================================
   HEADER-SPECIFIC CLASSES
============================================================================ */
.sg-header-wrap{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.25rem;
    flex-wrap: wrap;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.4rem;
    border-radius: var(--sg-radius);
    background: linear-gradient(120deg, rgba(14,22,40,0.75), rgba(10,16,30,0.55));
    border: 1px solid var(--sg-panel-border);
    backdrop-filter: blur(22px) saturate(150%);
    box-shadow: 0 10px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative;
    overflow: hidden;
}
.sg-header-wrap::after{
    content:"";
    position: absolute;
    top: 0; left: -30%;
    width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(34,232,255,0.08), transparent);
    animation: sg-sweep 6s linear infinite;
    pointer-events: none;
}
@keyframes sg-sweep{
    0%{ left: -40%; }
    100%{ left: 120%; }
}

.sg-brand-block{ display:flex; align-items:center; gap: 0.9rem; min-width: 260px; }

.sg-logo-ring{
    width: 52px; height: 52px;
    border-radius: 50%;
    display:flex; align-items:center; justify-content:center;
    font-size: 1.5rem;
    background: radial-gradient(circle at 35% 30%, rgba(34,232,255,0.25), rgba(6,10,20,0.6));
    border: 1.5px solid var(--sg-cyan);
    box-shadow: var(--sg-glow-strong);
    flex-shrink: 0;
    animation: sg-pulse-ring 3.2s ease-in-out infinite;
}
@keyframes sg-pulse-ring{
    0%, 100%{ box-shadow: 0 0 14px rgba(34,232,255,0.35), 0 0 34px rgba(34,232,255,0.14); }
    50%{ box-shadow: 0 0 24px rgba(34,232,255,0.6), 0 0 60px rgba(34,232,255,0.3); }
}

.sg-title-group{ line-height: 1.15; }
.sg-title{
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 1.65rem;
    letter-spacing: 0.03em;
    margin: 0;
    background: linear-gradient(90deg, #ffffff 0%, var(--sg-cyan-soft) 45%, var(--sg-indigo) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 30px rgba(34,232,255,0.25);
}
.sg-subtitle{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--sg-text-dim);
    margin-top: 2px;
}
.sg-subtitle strong{ color: var(--sg-cyan-soft); font-weight: 600; }

.sg-gov-badge{
    display:flex; align-items:center; gap: 0.5rem;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.09);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--sg-text-dim);
    letter-spacing: 0.04em;
    white-space: nowrap;
}
.sg-gov-badge .sg-flag{ font-size: 0.95rem; }

.sg-status-block{
    display:flex; align-items:center; gap: 0.9rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.sg-clock{
    font-family: 'JetBrains Mono', monospace;
    background: rgba(6,10,20,0.7);
    border: 1px solid var(--sg-panel-border);
    border-radius: var(--sg-radius-sm);
    padding: 0.5rem 0.9rem;
    text-align: right;
    min-width: 168px;
}
.sg-clock .sg-clock-time{
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--sg-cyan-soft);
    letter-spacing: 0.05em;
    text-shadow: 0 0 12px rgba(34,232,255,0.4);
}
.sg-clock .sg-clock-date{
    font-size: 0.68rem;
    color: var(--sg-text-faint);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 1px;
}

.sg-live-badge{
    display:flex; align-items:center; gap: 0.45rem;
    padding: 0.45rem 0.85rem 0.45rem 0.65rem;
    border-radius: 999px;
    background: rgba(43, 255, 168, 0.08);
    border: 1px solid rgba(43, 255, 168, 0.45);
    box-shadow: 0 0 16px rgba(43, 255, 168, 0.2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--sg-green);
}
.sg-live-dot{
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--sg-green);
    box-shadow: 0 0 8px var(--sg-green), 0 0 16px var(--sg-green);
    animation: sg-live-pulse 1.4s ease-in-out infinite;
}
@keyframes sg-live-pulse{
    0%{ transform: scale(0.85); opacity: 0.7; box-shadow: 0 0 4px var(--sg-green); }
    50%{ transform: scale(1.15); opacity: 1; box-shadow: 0 0 14px var(--sg-green), 0 0 26px var(--sg-green); }
    100%{ transform: scale(0.85); opacity: 0.7; box-shadow: 0 0 4px var(--sg-green); }
}

/* ============================================================================
   SIDEBAR NAV CLASSES
============================================================================ */
.sg-sidebar-brand{
    display:flex; flex-direction:column; align-items:center;
    text-align:center;
    padding: 0.6rem 0.4rem 1.0rem 0.4rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid var(--sg-panel-border);
}
.sg-sidebar-logo{
    width: 46px; height: 46px;
    border-radius: 50%;
    display:flex; align-items:center; justify-content:center;
    font-size: 1.3rem;
    background: radial-gradient(circle at 35% 30%, rgba(34,232,255,0.25), rgba(6,10,20,0.6));
    border: 1.5px solid var(--sg-cyan);
    box-shadow: var(--sg-glow);
    margin-bottom: 0.5rem;
}
.sg-sidebar-name{
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--sg-text);
    letter-spacing: 0.03em;
}
.sg-sidebar-tag{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--sg-text-faint);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
}

.sg-nav-caption{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sg-text-faint);
    margin: 0.9rem 0.2rem 0.4rem 0.2rem;
}

/* Radio-based nav re-skin into a vertical button list */
section[data-testid="stSidebar"] div[role="radiogroup"]{
    gap: 4px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
    background: rgba(255,255,255,0.015);
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 8px 12px 8px 10px;
    margin: 0;
    transition: all 0.18s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{
    background: rgba(34,232,255,0.07);
    border-color: var(--sg-panel-border);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
    background: linear-gradient(90deg, rgba(34,232,255,0.16), rgba(34,232,255,0.02));
    border-color: var(--sg-panel-border-strong);
    box-shadow: inset 2px 0 0 var(--sg-cyan), 0 0 14px rgba(34,232,255,0.12);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p{
    color: var(--sg-text) !important;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
}
section[data-testid="stSidebar"] div[role="radiogroup"] input{
    accent-color: var(--sg-cyan);
}

.sg-sidebar-footer{
    margin-top: 1.2rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--sg-panel-border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    color: var(--sg-text-faint);
    letter-spacing: 0.05em;
    line-height: 1.6;
}
.sg-sidebar-footer .sg-dot-ok{
    display:inline-block; width:6px; height:6px; border-radius:50%;
    background: var(--sg-green);
    box-shadow: 0 0 6px var(--sg-green);
    margin-right: 6px;
}

/* ============================================================================
   RESPONSIVE BREAKPOINTS
============================================================================ */
@media (max-width: 1200px){
    .sg-header-wrap{ padding: 1rem 1.2rem; }
    .sg-title{ font-size: 1.4rem; }
}

@media (max-width: 900px){
    .sg-header-wrap{ flex-direction: column; align-items: flex-start; gap: 0.9rem; }
    .sg-status-block{ width: 100%; justify-content: space-between; }
    .sg-clock{ min-width: unset; flex: 1; }
}

@media (max-width: 640px){
    section.main > div.block-container,
    section[data-testid="stMain"] > div.stMainBlockContainer,
    div[data-testid="stMainBlockContainer"]{ padding-left: 0.7rem; padding-right: 0.7rem; }
    .sg-title{ font-size: 1.15rem; }
    .sg-subtitle{ font-size: 0.68rem; }
    .sg-gov-badge{ display:none; }
    .sg-logo-ring{ width: 42px; height: 42px; font-size: 1.2rem; }
    .sg-clock{ padding: 0.4rem 0.6rem; }
    .sg-clock .sg-clock-time{ font-size: 0.9rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )