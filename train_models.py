"""
================================================================================
SkyGuard AI — Machine Learning Model Training Pipeline
================================================================================
Fits and serializes:
  1. WeatherFeatureEngineer (lags, rolling moments, cyclical harmonics, thermodynamic proxies)
  2. WeatherIsolationForest
  3. DeepWeatherAutoencoder (PyTorch CUDA/CPU)
  4. TemporalResidualPredictor (Multi-output trajectory forecasting)
================================================================================
"""

import os
import sys
import argparse
import time
import json
import pickle
import pandas as pd
import numpy as np

from features.engineering import WeatherFeatureEngineer
from models.isolation_forest import WeatherIsolationForest
from models.autoencoder import DeepWeatherAutoencoder
from models.temporal_model import TemporalResidualPredictor
from anomaly_injection.taxonomy import CORE_VARIABLES


def train_pipeline(input_csv: str, output_dir: str, epochs: int = 15):
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 78)
    print("  [+] SKYGUARD AI -- ML & DEEP LEARNING MODEL TRAINING PIPELINE")
    print("=" * 78)

    if not os.path.exists(input_csv):
        print(f"[!] Error: Input dataset '{input_csv}' not found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Loading training dataset from: {input_csv}")
    t0 = time.time()
    df = pd.read_csv(input_csv)
    print(f"[+] Loaded {len(df):,} observations across {df['station_id'].nunique()} stations in {time.time() - t0:.2f}s.")

    # 1. Feature Engineering
    print("\n[*] Extracting meteorological and temporal features...")
    t_feat = time.time()
    engineer = WeatherFeatureEngineer()
    X_scaled = engineer.fit_transform(df)
    y_target = df[CORE_VARIABLES].values
    feat_time = time.time() - t_feat
    print(f"[+] Engineered {X_scaled.shape[1]} features for {X_scaled.shape[0]:,} samples in {feat_time:.2f}s.")

    # Save Feature Engineer
    fe_path = os.path.join(output_dir, "feature_engineer.pkl")
    with open(fe_path, "wb") as f:
        pickle.dump(engineer, f)
    print(f"[+] Saved Feature Engineer to: {fe_path}")

    # 2. Train Isolation Forest
    print("\n[*] Training Weather Isolation Forest (150 trees)...")
    t_if = time.time()
    iso_model = WeatherIsolationForest(n_estimators=150, contamination=0.05, random_state=42)
    iso_model.fit(X_scaled)
    if_time = time.time() - t_if
    iso_path = os.path.join(output_dir, "isolation_forest.pkl")
    iso_model.save(iso_path)
    print(f"[+] Isolation Forest trained in {if_time:.2f}s. Saved to: {iso_path}")

    # 3. Train PyTorch Deep Autoencoder
    print(f"\n[*] Training PyTorch Deep Autoencoder ({epochs} epochs, GPU/CPU)...")
    t_ae = time.time()
    ae_model = DeepWeatherAutoencoder(epochs=epochs, batch_size=256, learning_rate=1e-3)
    print(f"    - Execution Device: {ae_model.device}")
    ae_model.fit(X_scaled)
    ae_time = time.time() - t_ae
    ae_path = os.path.join(output_dir, "autoencoder.pt")
    ae_model.save(ae_path)
    print(f"[+] Autoencoder trained in {ae_time:.2f}s (Threshold MSE: {ae_model.threshold:.4f}). Saved to: {ae_path}")

    # 4. Train Temporal Residual Predictor
    print("\n[*] Training Temporal Residual Predictor (Multi-output Ridge)...")
    t_tp = time.time()
    tp_model = TemporalResidualPredictor(alpha=1.0, threshold_sigma=3.5)
    tp_model.fit(X_scaled, y_target)
    tp_time = time.time() - t_tp
    tp_path = os.path.join(output_dir, "temporal_predictor.pkl")
    tp_model.save(tp_path)
    print(f"[+] Temporal Residual Predictor trained in {tp_time:.2f}s. Saved to: {tp_path}")

    # 5. Metadata summary
    meta = {
        "input_csv": input_csv,
        "n_samples": len(df),
        "n_features": X_scaled.shape[1],
        "feature_names": engineer.feature_names,
        "ae_device": str(ae_model.device),
        "ae_threshold": ae_model.threshold,
        "if_threshold": iso_model.threshold,
        "residual_stds": tp_model.residual_stds,
        "training_time_seconds": {
            "features": round(feat_time, 2),
            "isolation_forest": round(if_time, 2),
            "autoencoder": round(ae_time, 2),
            "temporal_predictor": round(tp_time, 2),
            "total": round(time.time() - t0, 2),
        },
    }
    meta_path = os.path.join(output_dir, "training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  [+] ALL MODELS TRAINED SUCCESSFULLY IN {time.time() - t0:.2f}s")
    print(f"  [+] Model artifacts saved in: {output_dir}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SkyGuard AI Machine Learning Models")
    parser.add_argument("--input", type=str, default="sample_aws_weather_data.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/models")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    train_pipeline(args.input, args.output_dir, args.epochs)
