#!/usr/bin/env python3
"""
Mobile-Money Fraud Detection on PaySim — starter pipeline
CPP 4103 (Artificial Intelligence Programming), Assignment 2

WHAT THIS DOES
  1. Loads PaySim (or generates a small fake demo dataset with --demo)
  2. Exploratory analysis  -> figures F1-F3
  3. Feature engineering   -> balance-error features, one-hot type
  4. Trains: Logistic Regression (class-weighted), Random Forest,
             XGBoost (if installed), SMOTE + Logistic Regression (if installed)
  5. Evaluates with fraud-appropriate metrics -> results.csv, figures F4-F7

HOW TO RUN
  # Dry run on synthetic data (no download needed):
  python fraud_detection.py --demo

  # Real thing (download PaySim from Kaggle: search
  # "Synthetic Financial Datasets For Fraud Detection", rename csv):
  python fraud_detection.py --data paysim.csv

  # Faster experimentation on a random subsample of 1M rows:
  python fraud_detection.py --data paysim.csv --sample 1000000

DEPENDENCIES
  pip install pandas numpy scikit-learn matplotlib
  pip install xgboost imbalanced-learn        # optional but recommended

NOTES FOR THE WRITE-UP
  * Accuracy is meaningless here (~0.13% fraud): always report
    precision, recall, F1, ROC-AUC and especially PR-AUC (average precision).
  * SMOTE is applied INSIDE the pipeline, i.e. only ever fit on training
    folds -- applying it before the split would leak information.
  * nameOrig / nameDest are dropped: raw IDs would let a model memorize
    accounts instead of learning fraud patterns (overfitting/leakage).
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

# ---------- optional libraries (degrade gracefully) ----------
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
FIGDIR = "figures"

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
]


# =====================================================================
# Demo data generator (PaySim-shaped) so the pipeline can be dry-run
# =====================================================================
def make_demo_data(n=120_000, fraud_rate=0.0013, seed=RANDOM_STATE):
    """Small synthetic dataset with the same schema as PaySim and a
    learnable fraud pattern (large amounts that drain the origin account).
    Numbers produced in --demo mode are for TESTING THE CODE ONLY."""
    rng = np.random.default_rng(seed)
    types = rng.choice(
        ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
        size=n,
        p=[0.22, 0.35, 0.01, 0.34, 0.08],
    )
    step = rng.integers(1, 744, n)
    amount = np.round(rng.lognormal(mean=9.5, sigma=1.4, size=n), 2)
    oldbalanceOrg = np.round(
        np.maximum(0.0, rng.lognormal(10, 1.6, n) - amount * rng.random(n)), 2
    )

    is_fraud = np.zeros(n, dtype=int)
    eligible = np.where((types == "TRANSFER") | (types == "CASH_OUT"))[0]
    n_fraud = max(10, int(n * fraud_rate))
    fraud_idx = rng.choice(eligible, size=n_fraud, replace=False)
    is_fraud[fraud_idx] = 1

    # LEGIT pattern (mirrors real PaySim): origin can afford the amount and
    # bookkeeping is usually consistent (new = old - amount)
    oldbalanceOrg = np.round(amount + rng.lognormal(10, 1.6, n), 2)
    consistent = rng.random(n) < 0.7
    newbalanceOrig = np.where(
        consistent,
        oldbalanceOrg - amount,
        np.round(np.maximum(0.0, oldbalanceOrg - amount * rng.random(n)), 2),
    )

    # FRAUD pattern (mirrors real PaySim): large amount that drains the
    # origin account exactly to zero
    amount[fraud_idx] = np.round(rng.lognormal(12.2, 0.9, n_fraud), 2)
    oldbalanceOrg[fraud_idx] = amount[fraud_idx]
    newbalanceOrig[fraud_idx] = 0.0

    oldbalanceDest = np.round(rng.lognormal(9, 2, n), 2) * (types != "PAYMENT")
    newbalanceDest = np.round(
        oldbalanceDest + np.where(rng.random(n) < 0.7, amount, 0.0), 2
    )

    df = pd.DataFrame(
        {
            "step": step,
            "type": types,
            "amount": amount,
            "nameOrig": ["C%09d" % i for i in rng.integers(0, 10**9, n)],
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "nameDest": ["C%09d" % i for i in rng.integers(0, 10**9, n)],
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "isFraud": is_fraud,
            "isFlaggedFraud": ((types == "TRANSFER") & (amount > 200_000)).astype(int),
        }
    )
    return df


# =====================================================================
# EDA
# =====================================================================
def run_eda(df):
    os.makedirs(FIGDIR, exist_ok=True)
    n = len(df)
    n_fraud = int(df["isFraud"].sum())
    print("\n========== EXPLORATORY ANALYSIS ==========")
    print(f"Rows: {n:,}   Fraud: {n_fraud:,}  ({100*n_fraud/n:.4f}%)")
    print("\nFraud count by transaction type:")
    print(df.groupby("type")["isFraud"].agg(["count", "sum"]).rename(
        columns={"count": "transactions", "sum": "frauds"}))

    flagged = df["isFlaggedFraud"].sum()
    caught = int(((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)).sum())
    print(f"\nBuilt-in rule 'isFlaggedFraud' flags {flagged:,} rows, "
          f"catching {caught:,} of {n_fraud:,} frauds "
          f"(recall = {caught/max(n_fraud,1):.4f})  <-- your baseline to beat")

    # F1: class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["isFraud"].value_counts().sort_index()
    ax.bar(["Legit (0)", "Fraud (1)"], counts.values, color=["#4472C4", "#C00000"])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("F1. Class balance")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/F1_class_balance.png", dpi=150)
    plt.close(fig)

    # F2: fraud by type
    by_type = df[df["isFraud"] == 1]["type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(by_type.index, by_type.values, color="#C00000")
    ax.set_ylabel("Fraudulent transactions")
    ax.set_title("F2. Fraud occurs only in some transaction types")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/F2_fraud_by_type.png", dpi=150)
    plt.close(fig)

    # F3: amount distributions
    fig, ax = plt.subplots(figsize=(6, 4))
    legit_amt = df.loc[df["isFraud"] == 0, "amount"].clip(lower=1)
    fraud_amt = df.loc[df["isFraud"] == 1, "amount"].clip(lower=1)
    bins = np.logspace(0, np.log10(max(df["amount"].max(), 10)), 50)
    # NOTE: with log-spaced bins, density=True is dominated by the tiny
    # first bins and produces an unreadable plot. Normalizing each class
    # to fractions makes the two distributions directly comparable.
    ax.hist(legit_amt, bins=bins, alpha=0.6, label="Legit", color="#4472C4",
            weights=np.ones(len(legit_amt)) / max(len(legit_amt), 1))
    ax.hist(fraud_amt, bins=bins, alpha=0.6, label="Fraud", color="#C00000",
            weights=np.ones(len(fraud_amt)) / max(len(fraud_amt), 1))
    ax.set_xscale("log")
    ax.set_xlabel("Amount (log scale)")
    ax.set_ylabel("Fraction of class")
    ax.set_title("F3. Transaction amount: fraud vs legit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/F3_amount_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved F1-F3 to {FIGDIR}/")


# =====================================================================
# Feature engineering
# =====================================================================
def engineer_features(df, focus=True):
    """Returns X (features) and y (labels).
    focus=True keeps only TRANSFER & CASH_OUT rows (the only types
    where fraud occurs) -- discuss this modelling choice in Section 5."""
    d = df.copy()
    if focus:
        d = d[d["type"].isin(["TRANSFER", "CASH_OUT"])]

    # The classic PaySim trick: fraudulent bookkeeping doesn't add up.
    d["errorBalanceOrig"] = d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"]
    d["errorBalanceDest"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]

    X = pd.get_dummies(
        d[NUMERIC_FEATURES + ["type"]], columns=["type"], drop_first=False
    ).astype(float)
    y = d["isFraud"].astype(int)
    return X, y


# =====================================================================
# Models
# =====================================================================
def build_models(y_train):
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    models = {
        "LogReg (class-weighted)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000,
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        print("[note] xgboost not installed -- skipping (pip install xgboost)")
    if HAS_SMOTE:
        models["SMOTE + LogReg"] = ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", LogisticRegression(max_iter=1000,
                                           random_state=RANDOM_STATE)),
            ]
        )
    else:
        print("[note] imbalanced-learn not installed -- skipping SMOTE variant")
    return models


# =====================================================================
# Evaluation
# =====================================================================
def evaluate(models, X_train, X_test, y_train, y_test):
    os.makedirs(FIGDIR, exist_ok=True)
    rows, probas = [], {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_s = time.time() - t0
        p = model.predict_proba(X_test)[:, 1]
        probas[name] = p
        pred = (p >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, p),
                "pr_auc": average_precision_score(y_test, p),
                "train_seconds": round(train_s, 1),
            }
        )
        print(f"  trained {name:<24s} in {train_s:6.1f}s")

    results = pd.DataFrame(rows).set_index("model").round(4)
    results.to_csv("results.csv")
    print("\n========== RESULTS (threshold = 0.5) ==========")
    print(results.to_string())
    print("\n(Headline metric = pr_auc. Accuracy is intentionally omitted --"
          " explain why in Section 5.6.)")

    # F4: PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas.items():
        PrecisionRecallDisplay.from_predictions(y_test, p, name=name, ax=ax)
    ax.set_title("F4. Precision-Recall curves")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/F4_pr_curves.png", dpi=150)
    plt.close(fig)

    # F5: ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas.items():
        RocCurveDisplay.from_predictions(y_test, p, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("F5. ROC curves")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/F5_roc_curves.png", dpi=150)
    plt.close(fig)

    # F6: feature importances from the best tree model available
    tree_name = "XGBoost" if "XGBoost" in models else "Random Forest"
    tree_model = models[tree_name]
    importances = pd.Series(
        tree_model.feature_importances_, index=X_train.columns
    ).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    importances.tail(10).plot.barh(ax=ax, color="#4472C4")
    ax.set_title(f"F6. Top feature importances ({tree_name})")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/F6_feature_importance.png", dpi=150)
    plt.close(fig)

    # F7: confusion matrix for best model by PR-AUC
    best = results["pr_auc"].idxmax()
    pred_best = (probas[best] >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_best)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred legit", "Pred fraud"])
    ax.set_yticks([0, 1], ["True legit", "True fraud"])
    ax.set_title(f"F7. Confusion matrix — {best}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/F7_confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved F4-F7 to {FIGDIR}/ ; results table -> results.csv")
    print(f"Best model by PR-AUC: {best}")
    return results


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="PaySim fraud detection pipeline")
    ap.add_argument("--data", default="paysim.csv", help="path to PaySim CSV")
    ap.add_argument("--demo", action="store_true",
                    help="run on generated fake data (no download needed)")
    ap.add_argument("--sample", type=int, default=None,
                    help="random subsample of N rows for faster runs")
    ap.add_argument("--all-types", action="store_true",
                    help="keep all 5 transaction types (default: TRANSFER+CASH_OUT)")
    args = ap.parse_args()

    if args.demo:
        print(">>> DEMO MODE: synthetic PaySim-shaped data. "
              "Numbers are for testing the pipeline only -- use the real "
              "Kaggle dataset for the report. <<<")
        df = make_demo_data()
    else:
        if not os.path.exists(args.data):
            raise SystemExit(
                f"'{args.data}' not found.\nDownload PaySim from Kaggle "
                "(search: Synthetic Financial Datasets For Fraud Detection), "
                "rename the CSV to paysim.csv, or pass --data <path>. "
                "Or dry-run with --demo."
            )
        df = pd.read_csv(args.data)

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=RANDOM_STATE)
        print(f"Subsampled to {len(df):,} rows")

    run_eda(df)

    X, y = engineer_features(df, focus=not args.all_types)
    print(f"\nFeature matrix: {X.shape[0]:,} rows x {X.shape[1]} features "
          f"({'TRANSFER+CASH_OUT only' if not args.all_types else 'all types'})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  "
          f"(stratified; see Section 5.3 for the temporal alternative)")

    print("\n========== TRAINING ==========")
    models = build_models(y_train)
    evaluate(models, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
