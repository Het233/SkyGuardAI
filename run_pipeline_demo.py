"""
================================================================================
SkyGuard AI — Complete End-to-End Pipeline Demonstration
================================================================================
Demonstrates the full operational lifecycle of SkyGuard AI:
1. Multi-Tier Anomaly Detection (QC, Statistical, ML/DL, Spatial Consensus)
2. Event-vs-Fault Reasoning & Root-Cause Classification
3. Explainable AI (XAI) Diagnostic Alert Cards & Prescriptive SOPs
4. Sensor & Station Health Scoring (0–100) & Degradation Forecasting
5. Non-Destructive Spatial-Temporal Value Imputation & Correction
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

# Fix Windows console UTF-8 output
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from anomaly_injection.taxonomy import CORE_VARIABLES
from classification.root_cause import RootCauseClassifier, attach_detector_scores
from reasoning.event_vs_fault import EventVsFaultClassifier
from explainability.explainer import AnomalyExplainer
from explainability.diagnostic_card import DiagnosticCardGenerator
from health.health_score import SensorHealthTracker
from imputation.correction import MeteorologicalImputer


def run_pipeline(
    input_path: str,
    models_dir: str = "artifacts/models",
    output_dir: str = "artifacts",
    sample_limit: int = 15000,
):
    print("=" * 80)
    print("  SKYGUARD AI — END-TO-END PIPELINE DEMONSTRATION")
    print("=" * 80)

    if not os.path.exists(input_path):
        print(f"[-] Input file not found: {input_path}")
        return

    print(f"[*] Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # Slice across all stations evenly by filtering on first N timestamps
    unique_times = df["timestamp"].drop_duplicates().sort_values().values
    if len(df) > sample_limit and len(unique_times) > 0:
        n_timestamps = max(1, sample_limit // df["station_id"].nunique())
        selected_times = set(unique_times[:n_timestamps])
        print(f"[*] Subsampling to {n_timestamps} timestamps across all {df['station_id'].nunique()} stations (~{sample_limit} rows)...")
        df = df[df["timestamp"].isin(selected_times)].copy().reset_index(drop=True)

    total_rows = len(df)
    anom_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    print(f"[+] Ingested {total_rows:,} records across {df['station_id'].nunique()} stations.")
    print(f"[+] Total ground-truth anomalies in slice: {anom_count} ({anom_count / total_rows * 100:.2f}%)")

    # 1. Attach detector scores
    print("\n" + "-" * 60)
    print("  STEP 1: MULTI-TIER ANOMALY SCORING")
    print("-" * 60)
    df_scored = attach_detector_scores(df, models_dir=models_dir)
    print(f"[+] Computed composite detector scores (Mean: {df_scored['composite_score'].mean():.3f}, Max: {df_scored['composite_score'].max():.3f})")

    # 2. Event-vs-Fault Reasoning
    print("\n" + "-" * 60)
    print("  STEP 2: EVENT-VS-FAULT REASONING")
    print("-" * 60)
    reasoner = EventVsFaultClassifier()
    anom_indices = np.where(df_scored["is_anomaly"].values == 1)[0]
    reasoning_demo_results = []
    for idx in anom_indices[:5]:
        row = df_scored.iloc[idx]
        t = row["timestamp"]
        st = row["station_id"]
        sub_time = df_scored[(df_scored["timestamp"] == t) & (df_scored["station_id"] != st)]
        neighbor_list = [sub_time.iloc[i] for i in range(len(sub_time))]
        st_hist = df_scored[df_scored["station_id"] == st]
        res = reasoner.evaluate_observation(
            row=row,
            df_history=st_hist,
            neighbor_rows=neighbor_list,
        )
        reasoning_demo_results.append({
            "station_id": st,
            "timestamp": str(t),
            "p_genuine_event": res["p_genuine_event"],
            "p_sensor_fault": res["p_sensor_fault"],
            "decision": res["classification"],
            "evidence": res["evidence"],
        })
    print(f"[+] Evaluated Event-vs-Fault reasoning for sample anomalies.")
    print(f"    Sample Attribution: {reasoning_demo_results[0]['decision']} (P_fault={reasoning_demo_results[0]['p_sensor_fault']:.2f})")

    # 3. Root Cause Classification
    print("\n" + "-" * 60)
    print("  STEP 3: MULTICLASS ROOT-CAUSE ATTRIBUTION")
    print("-" * 60)
    rc_path = os.path.join(models_dir, "root_cause_classifier.pkl")
    if os.path.exists(rc_path):
        rc_model = RootCauseClassifier.load(rc_path)
        pred_causes, confidences = rc_model.predict(df_scored)
        df_scored["predicted_cause"] = pred_causes
        df_scored["cause_confidence"] = confidences
        print(f"[+] Loaded trained RootCauseClassifier. Classified {len(df_scored):,} rows.")
        top_anom_causes = pd.Series(pred_causes[anom_indices]).value_counts().head(5)
        for c_name, count in top_anom_causes.items():
            print(f"    • {c_name:<28}: {count} occurrences")
    else:
        print("[-] RootCauseClassifier artifact not found; using ground truth labels.")
        df_scored["predicted_cause"] = df_scored.get("anomaly_type", "ANOMALY")
        df_scored["cause_confidence"] = 0.90
        rc_model = None

    # 4. Explainable AI & Diagnostic Alert Cards
    print("\n" + "-" * 60)
    print("  STEP 4: EXPLAINABLE AI (XAI) & DIAGNOSTIC CARDS")
    print("-" * 60)
    explainer = AnomalyExplainer()
    sample_cards = []
    for idx in anom_indices[:3]:
        row = df_scored.iloc[idx]
        prev_row = df_scored.iloc[idx - 1] if idx > 0 else None
        t = row["timestamp"]
        st = row["station_id"]
        nbr_rows = df_scored[(df_scored["timestamp"] == t) & (df_scored["station_id"] != st)]

        exp = explainer.explain_instance(
            row=row,
            prev_row=prev_row,
            neighbor_rows=nbr_rows,
            root_cause_model=rc_model,
        )
        card = DiagnosticCardGenerator.generate_card(
            exp,
            severity="CRITICAL" if row.get("composite_score", 0.8) > 0.75 else "ANOMALY",
            composite_score=float(row.get("composite_score", 0.75)),
        )
        sample_cards.append(card)

    print(f"[+] Generated {len(sample_cards)} XAI Diagnostic Alert Cards.")
    print("\n" + "=" * 50)
    print("  PREVIEW: SAMPLE DIAGNOSTIC ALERT CARD")
    print("=" * 50)
    print(DiagnosticCardGenerator.format_markdown(sample_cards[0]))

    # 5. Sensor Health Scoring & Degradation Forecasting
    print("-" * 60)
    print("  STEP 5: SENSOR HEALTH SCORING (0-100) & RUL PROJECTION")
    print("-" * 60)
    tracker = SensorHealthTracker(rolling_window_hours=168)
    station_reports = []
    for st_id, st_df in df_scored.groupby("station_id"):
        rep = tracker.compute_station_health(st_df)
        station_reports.append(rep)
        print(f"[+] Station {st_id:<10}: Overall Score = {rep.overall_score:>5.1f}/100 [{rep.status.value:<9}] Urgency: {rep.urgency}")
        for v_name, s_rep in rep.sensor_health.items():
            rul_str = f"RUL: {s_rep.days_to_critical:.1f}d" if s_rep.days_to_critical is not None else "Stable"
            print(f"    - {v_name:<22}: {s_rep.score:>5.1f}/100 ({s_rep.status.value}) | {rul_str}")

    # 6. Non-Destructive Value Imputation & Correction
    print("\n" + "-" * 60)
    print("  STEP 6: NON-DESTRUCTIVE VALUE IMPUTATION")
    print("-" * 60)
    imputer = MeteorologicalImputer(k_neighbors=3)
    df_imputed = imputer.impute_dataframe(df_scored, score_threshold=0.42)

    imputed_count = (df_imputed["imputation_flag"] != "RAW_PASSTHROUGH").sum()
    print(f"[+] Processed {len(df_imputed):,} records.")
    print(f"[+] Imputed/Corrected observations: {imputed_count:,} ({imputed_count / len(df_imputed) * 100:.2f}%)")
    flag_dist = df_imputed["imputation_flag"].value_counts()
    for f_name, count in flag_dist.items():
        print(f"    • {f_name:<28}: {count:,}")

    # Compute fidelity against clean ground truth if available
    has_clean = all(f"clean_{v}" in df_imputed.columns for v in CORE_VARIABLES)
    imputation_errors = {}
    if has_clean and imputed_count > 0:
        anom_mask = df_imputed["imputation_flag"] != "RAW_PASSTHROUGH"
        for v in CORE_VARIABLES:
            raw_err = np.nanmean(np.abs(df_imputed.loc[anom_mask, v] - df_imputed.loc[anom_mask, f"clean_{v}"]))
            imp_err = np.nanmean(np.abs(df_imputed.loc[anom_mask, f"imputed_{v}"] - df_imputed.loc[anom_mask, f"clean_{v}"]))
            imputation_errors[v] = {
                "raw_mae": round(float(raw_err), 3),
                "imputed_mae": round(float(imp_err), 3),
                "error_reduction_pct": round(float((raw_err - imp_err) / max(raw_err, 1e-3) * 100.0), 1),
            }
            print(f"[+] {v:<22}: Raw MAE = {raw_err:>6.2f} -> Imputed MAE = {imp_err:>6.2f} ({imputation_errors[v]['error_reduction_pct']}% error reduction)")

    # Save summary report and JSON artifacts
    def json_serializer(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    os.makedirs(output_dir, exist_ok=True)
    report_json = {
        "total_observations": total_rows,
        "anomalies_detected": int(imputed_count),
        "reasoning_sample": reasoning_demo_results,
        "sample_diagnostic_cards": sample_cards,
        "station_health_summaries": [rep.to_dict() for rep in station_reports],
        "imputation_performance": imputation_errors,
    }
    json_path = os.path.join(output_dir, "pipeline_demo_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, default=json_serializer)
    print(f"\n[+] Saved complete pipeline results to: {json_path}")

    # Write Markdown summary report
    md_path = os.path.join(output_dir, "pipeline_demo_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SkyGuard AI — End-to-End Pipeline Demonstration Report\n\n")
        f.write(f"- **Total Ingested Observations:** `{total_rows:,}`\n")
        f.write(f"- **Ground-Truth Anomalies:** `{anom_count:,}`\n")
        f.write(f"- **Imputed / Corrected Records:** `{imputed_count:,}`\n\n")
        f.write("## 1. Station Health Status Overview\n\n")
        f.write("| Station ID | Overall Health (0-100) | Status | Lowest Sensor | Operational Urgency |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for rep in station_reports:
            f.write(f"| `{rep.station_id}` | **{rep.overall_score:.1f}** | `{rep.status.value}` | `{rep.lowest_sensor}` | `{rep.urgency}` |\n")

        f.write("\n---\n\n## 2. Non-Destructive Imputation Performance\n\n")
        f.write("| Meteorological Parameter | Corrupted Raw MAE | Imputed Corrected MAE | Error Reduction |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for v, stats in imputation_errors.items():
            f.write(f"| `{v}` | {stats['raw_mae']} | **{stats['imputed_mae']}** | **{stats['error_reduction_pct']}%** |\n")

        f.write("\n---\n\n## 3. Sample Diagnostic Alert Card\n\n")
        f.write(DiagnosticCardGenerator.format_markdown(sample_cards[0]))

    print(f"[+] Saved Markdown summary report to: {md_path}")
    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkyGuard AI Pipeline Demonstration")
    parser.add_argument("--input", default="data/synthetic/sample_synthetic_anomalies.csv", help="Input dataset path")
    parser.add_argument("--models-dir", default="artifacts/models", help="Trained models directory")
    parser.add_argument("--output-dir", default="artifacts", help="Artifacts output directory")
    parser.add_argument("--limit", type=int, default=15000, help="Row limit for demonstration")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        sample_limit=args.limit,
    )
