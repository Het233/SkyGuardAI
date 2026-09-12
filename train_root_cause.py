"""
================================================================================
SkyGuard AI — Root-Cause Classifier Training Script
================================================================================
Trains a supervised multiclass Random Forest model to classify anomaly fault types
(Spike, Drop, Frozen Sensor, Drift, Offset, Noise, Missing Data, Corruption, etc.).
================================================================================
"""

import os
import sys
import argparse
import time
import pickle
import pandas as pd
import numpy as np

from features.engineering import WeatherFeatureEngineer
from models.isolation_forest import WeatherIsolationForest
from models.autoencoder import DeepWeatherAutoencoder
from models.temporal_model import TemporalResidualPredictor
from baseline.quality_control import DeterministicQCValidator
from baseline.statistical import CompositeBaselineDetector
from classification.root_cause import RootCauseClassifier, attach_detector_scores
from anomaly_injection.taxonomy import CORE_VARIABLES


def train_root_cause_model(csv_path: str, models_dir: str, n_estimators: int = 100):
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 78)
    print("  [+] SKYGUARD AI -- ROOT-CAUSE CLASSIFIER TRAINING")
    print("=" * 78)

    if not os.path.exists(csv_path):
        print(f"[!] Error: Dataset '{csv_path}' not found.")
        sys.exit(1)

    print(f"[*] Loading training dataset from: {csv_path}")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df):,} observations in {time.time() - t0:.2f}s.")

    # Attach detector scores
    print("[*] Computing detector score features for rich classifier inputs...")
    df = attach_detector_scores(df, models_dir)
    print("[+] Attached detector scores.")

    print(f"\n[*] Fitting RootCauseClassifier ({n_estimators} trees, balanced class weights)...")
    t1 = time.time()
    classifier = RootCauseClassifier(n_estimators=n_estimators, random_state=42)
    classifier.fit(df, target_col="anomaly_type")
    train_time = time.time() - t1
    print(f"[+] Trained classifier in {train_time:.2f}s.")

    # Evaluate train accuracy
    preds, confs = classifier.predict(df)
    train_acc = np.mean(preds == df["anomaly_type"].values)
    print(f"[+] Training Accuracy: {train_acc * 100:.2f}% across {len(classifier.classes_)} classes.")

    # Save artifact
    out_path = os.path.join(models_dir, "root_cause_classifier.pkl")
    classifier.save(out_path)
    print(f"[+] Saved Root-Cause Classifier to: {out_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Root Cause Classifier")
    parser.add_argument("--input", type=str, default="data/synthetic/sample_synthetic_anomalies.csv")
    parser.add_argument("--models-dir", type=str, default="artifacts/models")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    train_root_cause_model(args.input, args.models_dir, args.n_estimators)
