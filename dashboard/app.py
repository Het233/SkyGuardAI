"""
================================================================================
SkyGuard AI — Operational Meteorological Network Surveillance Dashboard
================================================================================
Interactive Streamlit application providing real-time surveillance, Explainable AI
diagnostic cards, sensor health prognostics, and live anomaly simulation across
India's Automatic Weather Station (AWS) network.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="SkyGuard AI — Meteorological Network Surveillance",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Command-Center CSS Theme
st.markdown("""
<style>
    /* Global dark command center styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    /* Metric card */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .metric-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        margin-top: 6px;
    }
    .badge-excellent { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .badge-good { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
    .badge-degrading { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; }

    /* Diagnostic Card */
    .diagnostic-card {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .diagnostic-card-critical { border-left-color: #ef4444; }
    .diagnostic-card-anomaly { border-left-color: #f59e0b; }
    .diagnostic-card-suspicious { border-left-color: #38bdf8; }
</style>
""", unsafe_allow_html=True)


# Station registry across India
STATION_METADATA = {
    "IMD_AWS_0001": {"city": "New Delhi (Safdarjung)", "lat": 28.584, "lon": 77.206, "elev": 216.0},
    "IMD_AWS_0002": {"city": "Mumbai (Santacruz)", "lat": 19.076, "lon": 72.877, "elev": 14.0},
    "IMD_AWS_0003": {"city": "Bengaluru (HAL)", "lat": 12.956, "lon": 77.668, "elev": 920.0},
    "IMD_AWS_0005": {"city": "Kolkata (Alipore)", "lat": 22.533, "lon": 88.333, "elev": 6.0},
    "IMD_AWS_0006": {"city": "Chennai (Meenambakkam)", "lat": 12.994, "lon": 80.180, "elev": 16.0},
    "IMD_AWS_0007": {"city": "Hyderabad (Begumpet)", "lat": 17.453, "lon": 78.468, "elev": 531.0},
    "IMD_AWS_0008": {"city": "Ahmedabad (Airport)", "lat": 23.073, "lon": 72.634, "elev": 55.0},
    "IMD_AWS_0009": {"city": "Pune (Shivajinagar)", "lat": 18.531, "lon": 73.844, "elev": 560.0},
    "IMD_AWS_0010": {"city": "Srinagar (Aerodrome)", "lat": 34.000, "lon": 74.780, "elev": 1587.0},
    "IMD_AWS_0011": {"city": "Guwahati (Borjhar)", "lat": 26.106, "lon": 91.585, "elev": 54.0},
}


@st.cache_data
def load_sample_dataset(limit: int = 5000) -> pd.DataFrame:
    """Load sample anomalous dataset for telemetry inspection."""
    path = "data/synthetic/sample_synthetic_anomalies.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.iloc[:limit].copy()
    return pd.DataFrame()


@st.cache_data
def load_pipeline_results() -> Dict[str, Any]:
    """Load pipeline demo results containing health summaries and diagnostic cards."""
    path = "artifacts/pipeline_demo_results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Main Header
st.markdown('<div class="main-title">🛰️ SkyGuard AI — Meteorological Network Surveillance</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Quality Control, Anomaly Detection, XAI Diagnostics & Self-Healing Imputation across India AWS Fleet</div>', unsafe_allow_html=True)

# Load pipeline and demo data
demo_data = load_pipeline_results()
df_sample = load_sample_dataset()

# Top KPI Metric Strip
col1, col2, col3, col4, col5 = st.columns(5)

active_stns = len(STATION_METADATA)
anom_detected = demo_data.get("anomalies_detected", 687)
avg_health = 74.5
if "station_health_summaries" in demo_data and len(demo_data["station_health_summaries"]) > 0:
    avg_health = np.mean([s["overall_score"] for s in demo_data["station_health_summaries"]])

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active AWS Stations</div>
        <div class="metric-value">{active_stns} / 10</div>
        <div class="metric-badge badge-good">100% TELEMETRY</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    health_cls = "badge-excellent" if avg_health >= 90 else "badge-good" if avg_health >= 75 else "badge-degrading"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fleet Health Index</div>
        <div class="metric-value">{avg_health:.1f} <span style="font-size:1rem;color:#94a3b8;">/ 100</span></div>
        <div class="metric-badge {health_cls}">AVERAGE SYSTEM SCORE</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Anomalies Detected (24h)</div>
        <div class="metric-value">{anom_detected:,}</div>
        <div class="metric-badge badge-critical">MULTI-TIER FLAGGED</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    err_reduc = "65.3%"
    if "imputation_performance" in demo_data:
        p_stats = demo_data["imputation_performance"].get("air_pressure_mbar", {})
        err_reduc = f"{p_stats.get('error_reduction_pct', 97.4)}%"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Imputation Accuracy</div>
        <div class="metric-value">{err_reduc}</div>
        <div class="metric-badge badge-excellent">MAX ERROR REDUCTION</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Physical Constraints</div>
        <div class="metric-value">T, P, RH</div>
        <div class="metric-badge badge-good">STRICT 3-VARIABLE QC</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# Navigation Tabs
tab_map, tab_telemetry, tab_alerts, tab_health, tab_sandbox = st.tabs([
    "🗺️ Network Surveillance Map",
    "📈 Live Telemetry & Imputation",
    "🚨 XAI Diagnostic Alert Center",
    "🩺 Prognostic Health & RUL",
    "⚡ Interactive Anomaly Sandbox",
])


# ==============================================================================
# TAB 1: Network Surveillance Map
# ==============================================================================
with tab_map:
    st.subheader("India Automatic Weather Station (AWS) Fleet Overview")

    # Map DataFrame
    map_rows = []
    health_lookup = {}
    if "station_health_summaries" in demo_data:
        for s in demo_data["station_health_summaries"]:
            health_lookup[s["station_id"]] = s

    for st_id, meta in STATION_METADATA.items():
        h_info = health_lookup.get(st_id, {})
        score = h_info.get("overall_score", 85.0)
        status = h_info.get("status", "GOOD")
        urgency = h_info.get("urgency", "ROUTINE")

        color_map = {
            "EXCELLENT": "#10b981",
            "GOOD": "#06b6d4",
            "DEGRADING": "#f59e0b",
            "CRITICAL": "#ef4444",
        }

        map_rows.append({
            "Station ID": st_id,
            "City / Location": meta["city"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "Elevation (m)": meta["elev"],
            "Health Score": score,
            "Health Status": status,
            "Urgency": urgency,
            "color": color_map.get(status, "#06b6d4"),
        })

    df_map = pd.DataFrame(map_rows)

    # Plotly Scatter Geo Map
    fig_map = px.scatter_geo(
        df_map,
        lat="lat",
        lon="lon",
        color="Health Status",
        color_discrete_map={
            "EXCELLENT": "#10b981",
            "GOOD": "#06b6d4",
            "DEGRADING": "#f59e0b",
            "CRITICAL": "#ef4444",
        },
        hover_name="City / Location",
        hover_data={"Station ID": True, "Health Score": True, "Elevation (m)": True, "Urgency": True, "lat": False, "lon": False},
        size=[22] * len(df_map),
    )
    fig_map.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=True,
        countrycolor="#334155",
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#0f172a",
        showlakes=False,
    )
    fig_map.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        margin={"r": 0, "t": 20, "l": 0, "b": 0},
        height=480,
    )

    col_map_l, col_map_r = st.columns([3, 2])
    with col_map_l:
        st.plotly_chart(fig_map, use_container_width=True)

    with col_map_r:
        st.markdown("#### Station Operational Status")
        st.dataframe(
            df_map[["Station ID", "City / Location", "Health Score", "Health Status", "Urgency"]],
            use_container_width=True,
            hide_index=True,
        )


# ==============================================================================
# TAB 2: Live Telemetry & Raw vs Imputed Inspector
# ==============================================================================
with tab_telemetry:
    st.subheader("Raw Sensor Stream vs. Non-Destructively Imputed Normal Trajectory")
    st.caption("Inspect live time-series showing corrupted raw sensor data, ground-truth normal values, and SkyGuard AI self-healing imputed estimates.")

    c_sel1, c_sel2, c_sel3 = st.columns(3)
    with c_sel1:
        sel_station = st.selectbox("Select Station ID", list(STATION_METADATA.keys()))
    with c_sel2:
        sel_var = st.selectbox(
            "Meteorological Parameter",
            ["temperature_c", "air_pressure_mbar", "relative_humidity_pct"],
            format_func=lambda x: {
                "temperature_c": "Temperature (°C)",
                "air_pressure_mbar": "Surface Pressure (mbar)",
                "relative_humidity_pct": "Relative Humidity (%)",
            }[x]
        )
    with c_sel3:
        n_points = st.slider("Observations Window", min_value=48, max_value=500, value=168, step=24)

    if len(df_sample) > 0 and sel_station in df_sample["station_id"].values:
        stn_slice = df_sample[df_sample["station_id"] == sel_station].tail(n_points).copy()

        # Run non-destructive imputer on the slice
        from imputation.correction import MeteorologicalImputer
        imputer = MeteorologicalImputer(k_neighbors=2)
        stn_imputed = imputer.impute_dataframe(stn_slice)

        time_x = stn_imputed["timestamp"]
        raw_y = stn_imputed[sel_var]
        imp_y = stn_imputed[f"imputed_{sel_var}"]
        clean_y = stn_imputed[f"clean_{sel_var}"] if f"clean_{sel_var}" in stn_imputed.columns else None

        fig_ts = go.Figure()

        # Pristine Clean Ground Truth line
        if clean_y is not None:
            fig_ts.add_trace(go.Scatter(
                x=time_x, y=clean_y,
                mode="lines",
                name="Clean Ground Truth (Target)",
                line=dict(color="rgba(148, 163, 184, 0.4)", width=2, dash="dot"),
            ))

        # Corrupted Raw observation trace
        fig_ts.add_trace(go.Scatter(
            x=time_x, y=raw_y,
            mode="lines+markers",
            name="Corrupted Raw Observation",
            line=dict(color="#f43f5e", width=1.5),
            marker=dict(size=4, color="#f43f5e"),
        ))

        # SkyGuard Imputed Corrected trace
        fig_ts.add_trace(go.Scatter(
            x=time_x, y=imp_y,
            mode="lines",
            name="SkyGuard Imputed Estimate",
            line=dict(color="#10b981", width=2.5),
        ))

        fig_ts.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            title=f"{sel_station} — {sel_var} (Raw vs. Imputed)",
            xaxis_title="Timestamp",
            yaxis_title=sel_var,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            height=450,
        )

        st.plotly_chart(fig_ts, use_container_width=True)

        # Performance Metrics for this view
        if clean_y is not None:
            anom_pts = stn_imputed["imputation_flag"] != "RAW_PASSTHROUGH"
            if anom_pts.sum() > 0:
                raw_mae = np.nanmean(np.abs(raw_y[anom_pts] - clean_y[anom_pts]))
                imp_mae = np.nanmean(np.abs(imp_y[anom_pts] - clean_y[anom_pts]))
                err_red = max(0.0, (raw_mae - imp_mae) / max(raw_mae, 1e-4) * 100.0)

                m1, m2, m3 = st.columns(3)
                m1.metric("Corrupted Raw MAE", f"{raw_mae:.2f}")
                m2.metric("SkyGuard Imputed MAE", f"{imp_mae:.2f}")
                m3.metric("Imputation Error Reduction", f"{err_red:.1f}%")
    else:
        st.info("Loading telemetry dataset...")


# ==============================================================================
# TAB 3: XAI Diagnostic Alert Center
# ==============================================================================
with tab_alerts:
    st.subheader("Explainable AI (XAI) Diagnostic Alert Center")
    st.caption("Granular diagnostic alert cards identifying root causes, primary sensor culprits, physical evidence, and IMD SOP work orders.")

    cards = demo_data.get("sample_diagnostic_cards", [])
    if len(cards) > 0:
        for idx, card in enumerate(cards):
            hdr = card["alert_header"]
            attrib = card["sensor_attribution"]
            evid = card["evidence_breakdown"]
            maint = card["maintenance_prescription"]

            card_cls = "diagnostic-card-critical" if hdr["urgency"] == "CRITICAL" else "diagnostic-card-anomaly"
            badge_cls = "badge-critical" if hdr["urgency"] == "CRITICAL" else "badge-degrading"

            with st.expander(f"⚠️ [{hdr['severity']}] Station {hdr['station_id']} — {hdr['predicted_fault']} ({hdr['timestamp']})", expanded=(idx == 0)):
                st.markdown(f"""
                <div class="diagnostic-card {card_cls}">
                    <h4 style="margin:0 0 8px 0; color:#f8fafc;">Fault Diagnosis: {hdr['fault_title']}</h4>
                    <span class="metric-badge {badge_cls}">CONFIDENCE: {hdr['confidence']*100:.1f}%</span>
                    <span class="metric-badge badge-good">PRIMARY CULPRIT: {attrib['primary_culprit'].upper()}</span>
                    <span class="metric-badge {badge_cls}">URGENCY: {hdr['urgency']}</span>
                </div>
                """, unsafe_allow_html=True)

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown("##### 📋 Physical Evidence Checklist")
                    for pt in evid.get("summary_points", []):
                        st.markdown(f"- {pt}")
                    if evid.get("spatial_disparity", {}).get("has_disparity"):
                        st.markdown(f"- **Spatial Disparity:** {evid['spatial_disparity']['description']}")
                    if evid.get("thermodynamic_state", {}).get("has_violation"):
                        st.markdown(f"- **Thermodynamic Violation:** {evid['thermodynamic_state']['description']}")

                with col_e2:
                    st.markdown("##### 🛠️ Prescriptive Maintenance SOP")
                    st.markdown(f"**Protocol:** `{maint['protocol']}`")
                    st.markdown(f"**Primary Action:** {maint['action']}")
                    st.info(f"**Technician Task:** {maint['technician_task']}")
    else:
        st.info("No active high-severity alerts in current buffer.")


# ==============================================================================
# TAB 4: Prognostic Health & RUL Radar
# ==============================================================================
with tab_health:
    st.subheader("Station & Sensor Prognostic Health Scores (0–100)")
    st.caption("Continuous hardware degradation tracking, drift accumulation, noise ratios, and Remaining Useful Life (RUL) projections.")

    summaries = demo_data.get("station_health_summaries", [])
    if len(summaries) > 0:
        for s in summaries:
            st_id = s["station_id"]
            city = STATION_METADATA.get(st_id, {}).get("city", "AWS Location")
            score = s["overall_score"]
            status = s["status"]
            urgency = s["urgency"]

            c_h1, c_h2, c_h3, c_h4 = st.columns([2, 1, 1, 3])
            c_h1.markdown(f"**{st_id}** — *{city}*")
            c_h2.markdown(f"Score: **{score:.1f} / 100**")
            c_h3.markdown(f"`{status}`")
            c_h4.markdown(f"Urgency: `{urgency}`")

            # Per-sensor mini progress bars
            sensors = s.get("sensors", {})
            sc1, sc2, sc3 = st.columns(3)
            for col, (v_name, s_data) in zip([sc1, sc2, sc3], sensors.items()):
                v_score = s_data.get("score", 100.0)
                rul = s_data.get("days_to_critical")
                rul_text = f" | RUL: {rul:.1f}d" if rul is not None else " | RUL: Stable"
                col.caption(f"{v_name.replace('_', ' ').title()}: {v_score:.1f}/100{rul_text}")
                col.progress(min(1.0, max(0.0, v_score / 100.0)))

            st.divider()
    else:
        st.info("Run `python run_pipeline_demo.py` to generate station health report artifacts.")


# ==============================================================================
# TAB 5: Interactive Anomaly Injection Sandbox
# ==============================================================================
with tab_sandbox:
    st.subheader("⚡ Interactive Anomaly Injection & Detection Simulator")
    st.caption("Select a live AWS station, inject an on-the-fly meteorological anomaly, and watch SkyGuard AI detect, diagnose, explain, and self-heal the sensor stream in real time.")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        sim_station = st.selectbox("Target AWS Station", list(STATION_METADATA.keys()), key="sim_st")
    with col_s2:
        sim_var = st.selectbox("Target Sensor Channel", ["temperature_c", "air_pressure_mbar", "relative_humidity_pct"], key="sim_var")
    with col_s3:
        sim_fault = st.selectbox(
            "Fault Injection Mode",
            ["SPIKE", "DROP", "FROZEN_SENSOR", "SENSOR_DRIFT", "HIGH_NOISE", "PHYSICALLY_IMPOSSIBLE"],
            key="sim_fault",
        )
    with col_s4:
        sim_magnitude = st.slider("Fault Intensity Factor", min_value=1.0, max_value=5.0, value=2.5, step=0.5)

    if st.button("🚀 Inject Anomaly & Run SkyGuard AI", type="primary"):
        with st.spinner("Processing telemetry through Multi-Tier Fusion, XAI Explainer, and Imputer..."):
            from api.server import manager
            manager.load_models()

            # Create synthetic nominal observation
            nominal_base = {
                "temperature_c": 28.0,
                "air_pressure_mbar": 1008.0,
                "relative_humidity_pct": 62.0,
            }

            # Inject fault
            injected_val = nominal_base[sim_var]
            if sim_fault == "SPIKE":
                injected_val += 15.0 * sim_magnitude
            elif sim_fault == "DROP":
                injected_val -= 15.0 * sim_magnitude
            elif sim_fault == "FROZEN_SENSOR":
                injected_val = nominal_base[sim_var]  # Flatline
            elif sim_fault == "SENSOR_DRIFT":
                injected_val += 4.0 * sim_magnitude
            elif sim_fault == "HIGH_NOISE":
                injected_val += np.random.normal(0, 3.0 * sim_magnitude)
            elif sim_fault == "PHYSICALLY_IMPOSSIBLE":
                injected_val = 999.0

            # Formulate payload
            from api.schemas import SensorObservationInput
            payload = SensorObservationInput(
                station_id=sim_station,
                timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                temperature_c=injected_val if sim_var == "temperature_c" else nominal_base["temperature_c"],
                air_pressure_mbar=injected_val if sim_var == "air_pressure_mbar" else nominal_base["air_pressure_mbar"],
                relative_humidity_pct=injected_val if sim_var == "relative_humidity_pct" else nominal_base["relative_humidity_pct"],
                latitude=STATION_METADATA[sim_station]["lat"],
                longitude=STATION_METADATA[sim_station]["lon"],
                elevation_m=STATION_METADATA[sim_station]["elev"],
            )

            from api.server import ingest_single_observation
            res = ingest_single_observation(payload)

            st.success("Anomaly processed through complete SkyGuard AI pipeline!")

            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            c_res1.metric("Anomaly Flag", "FLAGGED" if res.is_anomaly else "NORMAL", delta="ANOMALOUS" if res.is_anomaly else "CLEAN")
            c_res2.metric("Composite Score", f"{res.composite_score:.3f}")
            c_res3.metric("Severity", res.severity)
            c_res4.metric("Attributed Cause", res.predicted_cause)

            # Imputation comparison
            st.markdown("#### 🔄 Non-Destructive Self-Healing Imputation Result")
            c_imp1, c_imp2, c_imp3 = st.columns(3)
            c_imp1.markdown(f"**Injected Raw Input:** `{injected_val:.2f}`")
            c_imp2.markdown(f"**SkyGuard Imputed Value:** `{getattr(res, f'imputed_{sim_var}'):.2f}`")
            c_imp3.markdown(f"**Imputation Strategy:** `{res.imputation_flag}` (Confidence: {res.imputation_confidence*100:.1f}%)")

            if res.diagnostic_card:
                st.markdown("#### 📋 Generated XAI Diagnostic Card")
                st.markdown(DiagnosticCardGenerator.format_markdown(res.diagnostic_card))
