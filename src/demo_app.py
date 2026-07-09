#!/usr/bin/env python3
"""
Assignment 3 — Step 2: live fraud-checker demo (Streamlit).

Run AFTER train_model.py has produced fraud_model.joblib, from the
same folder:

  pip install streamlit
  streamlit run demo_app.py

A browser tab opens automatically. Project it in class, use the preset
buttons for a smooth demo, and drag the threshold slider to show the
precision/recall trade-off live.
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Mobile-Money Fraud Checker", page_icon="🛡️",
                   layout="centered")


# ---------- scoring logic (kept pure so it is easy to test) ----------
def make_feature_row(feature_names, step, tx_type, amount,
                     old_org, new_org, old_dest, new_dest):
    vals = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "errorBalanceOrig": new_org + amount - old_org,
        "errorBalanceDest": old_dest + amount - new_dest,
        "type_CASH_OUT": 1.0 if tx_type == "CASH_OUT" else 0.0,
        "type_TRANSFER": 1.0 if tx_type == "TRANSFER" else 0.0,
    }
    return pd.DataFrame([[float(vals.get(f, 0.0)) for f in feature_names]],
                        columns=feature_names)


@st.cache_resource
def load_bundle():
    return joblib.load("fraud_model.joblib")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("fraud_model.joblib not found. Run train_model.py first, "
             "in this same folder.")
    st.stop()

model = bundle["model"]
feature_names = bundle["feature_names"]

# ---------- header ----------
st.title("🛡️ Mobile-Money Fraud Checker")
st.caption(
    f"{bundle['model_name']} trained on {bundle['trained_rows']:,} PaySim "
    f"transactions · holdout PR-AUC {bundle['metrics_holdout']['pr_auc']} · "
    f"recall {bundle['metrics_holdout']['recall']:.1%} · "
    f"precision {bundle['metrics_holdout']['precision']:.1%}"
)
if bundle.get("demo_mode"):
    st.warning("This model was trained in --demo mode on fake data. "
               "Retrain on the real PaySim CSV before presenting.")

# ---------- sidebar ----------
st.sidebar.header("Decision threshold")
threshold = st.sidebar.slider(
    "Flag as fraud when probability ≥", 0.01, 0.99, 0.50, 0.01)
st.sidebar.markdown(
    "Lower it → catch more fraud but raise false alarms. "
    "Higher it → fewer alarms but more fraud slips through. "
    "This is the cost trade-off from our report (Section 7).")

# ---------- presets ----------
PRESETS = {
    "typical legit": dict(step=250, tx_type="CASH_OUT", amount=12_500.0,
                          old_org=85_000.0, new_org=72_500.0,
                          old_dest=140_000.0, new_dest=152_500.0),
    "fraud pattern": dict(step=300, tx_type="TRANSFER", amount=850_000.0,
                          old_org=850_000.0, new_org=0.0,
                          old_dest=0.0, new_dest=0.0),
}
c1, c2 = st.columns(2)
if c1.button("Load a typical LEGITIMATE transaction", use_container_width=True):
    st.session_state.update(PRESETS["typical legit"])
if c2.button("Load the classic FRAUD pattern", use_container_width=True):
    st.session_state.update(PRESETS["fraud pattern"])

# ---------- inputs ----------
st.subheader("Transaction details")
i1, i2 = st.columns(2)
tx_type = i1.selectbox("Type", ["TRANSFER", "CASH_OUT"],
                       index=["TRANSFER", "CASH_OUT"].index(
                           st.session_state.get("tx_type", "TRANSFER")))
step = i2.number_input("Hour of month (step)", 1, 743,
                       int(st.session_state.get("step", 300)))
amount = st.number_input("Amount", min_value=0.0,
                         value=float(st.session_state.get("amount", 850_000.0)),
                         step=1000.0, format="%.2f")
b1, b2 = st.columns(2)
old_org = b1.number_input("Origin balance BEFORE", min_value=0.0,
                          value=float(st.session_state.get("old_org", 850_000.0)),
                          step=1000.0, format="%.2f")
new_org = b2.number_input("Origin balance AFTER", min_value=0.0,
                          value=float(st.session_state.get("new_org", 0.0)),
                          step=1000.0, format="%.2f")
b3, b4 = st.columns(2)
old_dest = b3.number_input("Destination balance BEFORE", min_value=0.0,
                           value=float(st.session_state.get("old_dest", 0.0)),
                           step=1000.0, format="%.2f")
new_dest = b4.number_input("Destination balance AFTER", min_value=0.0,
                           value=float(st.session_state.get("new_dest", 0.0)),
                           step=1000.0, format="%.2f")

# engineered features, shown live (nice teaching moment)
err_orig = new_org + amount - old_org
err_dest = old_dest + amount - new_dest
st.caption(f"Engineered features → errorBalanceOrig = {err_orig:,.2f} · "
           f"errorBalanceDest = {err_dest:,.2f} "
           f"(fraudulent bookkeeping usually doesn't add up)")

# ---------- predict ----------
if st.button("🔍 Score this transaction", type="primary",
             use_container_width=True):
    row = make_feature_row(feature_names, step, tx_type, amount,
                           old_org, new_org, old_dest, new_dest)
    proba = float(model.predict_proba(row)[0, 1])
    st.progress(min(max(proba, 0.0), 1.0),
                text=f"Fraud probability: {proba:.1%}")
    if proba >= threshold:
        st.error(f"🚨 FRAUD ALERT — probability {proba:.1%} ≥ "
                 f"threshold {threshold:.0%}. Recommend blocking for review.")
    else:
        st.success(f"✅ Looks legitimate — probability {proba:.1%} < "
                   f"threshold {threshold:.0%}.")

st.divider()
st.caption("Educational demo for CPP 4103 Assignment 3. Model trained on "
           "the synthetic PaySim dataset; not a production fraud system.")
