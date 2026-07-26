# Mobile-Money Fraud Detection — CPP 4103 Assignment 3

**KCA University · Artificial Intelligence Programming (CPP 4103)**
**Name:** Adrian Njunge
**Registration Number:** 23/03607

An artificial-intelligence solution that detects fraudulent mobile-money
transactions, developed for the problem identified in Assignment 2. This
repository contains the source code, the trained-model results, and a live
demo application.

---

## 1. Problem

Mobile money is central to everyday finance in Kenya and much of Africa, which
makes transaction fraud a costly, high-stakes problem. Fraud is extremely rare
(about **0.13%** of transactions), so the challenge is to detect it reliably
without drowning genuine customers in false alarms.

## 2. Dataset

**PaySim** — a public dataset that simulates one month (≈6.36 million
transactions) of a real African mobile-money operator, published on Kaggle as
*"Synthetic Financial Datasets For Fraud Detection."* It is synthetic, so it
contains no private data. The CSV (~470 MB) is **not** included in this repo;
download it from Kaggle and place it next to the scripts (see below).

## 3. Solution / AI method

- **Feature engineering:** "balance-error" features that capture when a
  transaction's bookkeeping does not add up — the fingerprint of fraud.
- **Models compared:** Logistic Regression, Random Forest, XGBoost, and
  SMOTE + Logistic Regression, all with explicit class-imbalance handling.
- **Honest evaluation:** an 80/20 stratified split judged on **Precision–Recall**
  (PR-AUC), not accuracy, because accuracy is meaningless at 0.13% fraud.
- **Deployment:** the best model is saved and served through an interactive
  fraud-checker web app.

## 4. Key results

On a held-out test set of 554,082 transactions (full dataset):

| Model | Precision | Recall | PR-AUC |
|---|---|---|---|
| Logistic Regression | 4.3% | 88.4% | 0.587 |
| **Random Forest** | **100%** | **99.6%** | **0.998** |
| XGBoost | 99.2% | 99.7% | 0.998 |
| SMOTE + Logistic Regression | 4.3% | 89.4% | 0.584 |

**Headline:** the Random Forest caught **1,637 of 1,643 frauds with zero false
alarms** — versus the operator's built-in rule, which caught just 16 of 8,213
frauds (0.19% recall). Full numbers are in [`results/results.csv`](results/results.csv);
figures are in [`results/figures/`](results/figures).

A key finding (see the write-up): the linear models score **ROC-AUC ≈ 0.98**
yet collapse to **PR-AUC ≈ 0.59** — a clear demonstration that ROC-AUC flatters
classifiers on imbalanced data, while PR-AUC tells the truth.

## 5. Repository structure

```
├── README.md                ← this file
├── requirements.txt         ← Python dependencies
├── src/
│   ├── fraud_detection.py   ← full pipeline: EDA, training, evaluation, figures
│   ├── train_model.py       ← trains & saves the deployable model
│   ├── demo_app.py          ← live fraud-checker web app (Streamlit)
│   └── fix_f3.py            ← regenerates one corrected figure
├── results/
│   ├── results.csv          ← model comparison metrics
│   └── figures/             ← F1–F7 (class balance, curves, importances, etc.)
└── docs/
    └── RUN_OF_SHOW.md        ← presentation guide
```

## 6. How to run

**Prerequisites:** Python 3.10+ and the PaySim CSV from Kaggle.

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. reproduce the full analysis (EDA + all models + figures)
python src/fraud_detection.py --data path/to/paysim.csv

# 3. train and save the deployable model  ->  fraud_model.joblib
python src/train_model.py --data path/to/paysim.csv

# 4. launch the live fraud-checker demo (opens in your browser)
pip install streamlit
streamlit run src/demo_app.py
```

**No dataset yet?** Every script runs on generated sample data with `--demo`,
e.g. `python src/train_model.py --demo`, so the pipeline can be tested without
the 470 MB download. (Demo-mode numbers are illustrative only.)

## 7. Notes & limitations

- PaySim is **synthetic**: its fraud follows a consistent account-draining rule
  that our engineered feature captures almost perfectly, so the near-perfect
  scores would be lower on live data, where fraudsters adapt.
- A production system would need temporal (train-on-past, test-on-future)
  validation and human review before any transaction is blocked.
- This is an educational project, not a production fraud system.

## 8. Acknowledgements

Dataset: Lopez-Rojas, Elmir & Axelsson (2016), *PaySim: A financial mobile
money simulator for fraud detection*, EMSS. Built with scikit-learn, XGBoost,
imbalanced-learn, pandas, matplotlib and Streamlit.
