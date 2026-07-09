#!/usr/bin/env python3
"""
Assignment 3 — Step 1: train and SAVE the deployable fraud model.

Must sit in the SAME folder as fraud_detection.py (it reuses the same
feature engineering, so the demo app scores transactions exactly the
way the report's models were trained).

Usage:
  python train_model.py --data "C:\\Users\\njung\\Downloads\\archive\\PS_20174392719_1491204439457_log.csv"
  python train_model.py --demo          # quick dry-run on fake data

Output: fraud_model.joblib  (model + feature order + test metrics)

Default model is XGBoost: on our Assignment 2 run it matched the Random
Forest (PR-AUC 0.998 vs 0.998, recall 99.7% vs 99.6%) while training
6.5x faster and producing a model file of a few MB instead of hundreds
-- the right trade-off for deployment. Use --model rf to save the
Random Forest instead.
"""

import argparse
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

from fraud_detection import engineer_features, make_demo_data, RANDOM_STATE


def build(model_name, y_train):
    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE)
    from xgboost import XGBClassifier
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=neg / max(pos, 1), eval_metric="aucpr",
        n_jobs=-1, random_state=RANDOM_STATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="paysim.csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    ap.add_argument("--out", default="fraud_model.joblib")
    args = ap.parse_args()

    df = make_demo_data() if args.demo else pd.read_csv(args.data)
    if args.demo:
        print(">>> DEMO MODE: model trained on fake data -- for testing "
              "the app only. Retrain on the real CSV before presenting. <<<")

    X, y = engineer_features(df, focus=True)
    feature_names = list(X.columns)
    print(f"Data: {len(X):,} rows x {len(feature_names)} features")

    # 1) honest holdout evaluation
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model = build(args.model, y_tr)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, pred, zero_division=0), 4),
        "f1": round(f1_score(y_te, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_te, p), 4),
        "pr_auc": round(average_precision_score(y_te, p), 4),
    }
    print("Holdout metrics:", metrics)

    # 2) refit on ALL data for the deployed model
    final = build(args.model, y)
    final.fit(X, y)
    print(f"Trained final model on all rows in {time.time()-t0:.1f}s total")

    bundle = {
        "model": final,
        "model_name": "XGBoost" if args.model == "xgb" else "Random Forest",
        "feature_names": feature_names,
        "metrics_holdout": metrics,
        "trained_rows": int(len(X)),
        "demo_mode": bool(args.demo),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, args.out, compress=3)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
