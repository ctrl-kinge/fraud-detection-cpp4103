#!/usr/bin/env python3
"""
Regenerates ONLY Figure F3 (amount distribution) with corrected
normalization -- no model training, reads just 2 columns, runs in ~1 min.

Usage:
  python fix_f3.py --data "C:\\Users\\njung\\Downloads\\archive\\PS_20174392719_1491204439457_log.csv"

Writes: figures/F3_amount_distribution.png
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="paysim.csv")
args = ap.parse_args()

print("Reading amount + isFraud columns only ...")
df = pd.read_csv(args.data, usecols=["amount", "isFraud"])
os.makedirs("figures", exist_ok=True)

legit = df.loc[df["isFraud"] == 0, "amount"].clip(lower=1)
fraud = df.loc[df["isFraud"] == 1, "amount"].clip(lower=1)
bins = np.logspace(0, np.log10(max(df["amount"].max(), 10)), 50)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(legit, bins=bins, alpha=0.6, label="Legit", color="#4472C4",
        weights=np.ones(len(legit)) / len(legit))
ax.hist(fraud, bins=bins, alpha=0.6, label="Fraud", color="#C00000",
        weights=np.ones(len(fraud)) / len(fraud))
ax.set_xscale("log")
ax.set_xlabel("Amount (log scale)")
ax.set_ylabel("Fraction of class")
ax.set_title("F3. Transaction amount: fraud vs legit")
ax.legend()
fig.tight_layout()
fig.savefig("figures/F3_amount_distribution.png", dpi=150)
print("Saved figures/F3_amount_distribution.png -- done.")
