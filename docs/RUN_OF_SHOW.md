# Assignment 3 — Run-of-Show Guide

**Task (from the brief):** apply/develop an AI solution to the problem from Assignment 2, **present it in class**, then upload.
Our problem: detecting mobile-money fraud in PaySim. Our solution: a trained model + a live fraud-checker app.

You have three deliverables in this kit:
1. `train_model.py` — trains and saves the deployable model (`fraud_model.joblib`)
2. `demo_app.py` — the live fraud-checker you show in class
3. `A3_Fraud_Detection_Presentation.pptx` — the 10-slide deck (with speaker notes)

---

## STEP 1 — Train the deployable model (once, ~1–3 min)

Put `train_model.py` in the **same folder** as `fraud_detection.py` (it reuses the same feature engineering). Then:

```
python "C:\Users\njung\Downloads\train_model.py" --data "C:\Users\njung\Downloads\archive\PS_20174392719_1491204439457_log.csv"
```

This prints the holdout metrics and writes **`fraud_model.joblib`** (a few MB) to your current folder. That file *is* your deployed AI — the app loads it.

*Dry run first if you like:* `python train_model.py --demo` (uses fake data, for testing only — retrain on the real CSV before presenting).

## STEP 2 — Launch the live demo (test it the night before!)

```
pip install streamlit
streamlit run "C:\Users\njung\Downloads\demo_app.py"
```

A browser tab opens at `http://localhost:8501`. Keep `demo_app.py` and `fraud_model.joblib` in the **same folder**. Leave this tab open during your talk.

**If Streamlit won't cooperate on the presentation laptop:** run it once at home, screen-record the two presets + the slider (60 seconds), and play that video from slide 8 as a backup. Always have this backup.

---

## Presentation flow (~12–14 min, fits the class slot)

| Slide | Who | Time | Key line to land |
|---|---|---|---|
| 1 Title | All | 0:30 | "Mobile money runs Kenya's economy — fraud in it is a high-stakes AI problem." |
| 2 Problem | Speaker 1 | 1:30 | One fraud per ~775 transactions; missing one hurts a customer. |
| 3 Data | Speaker 1/2 | 1:30 | Synthetic but realistic; fraud only in TRANSFER/CASH_OUT. |
| 4 Method | Speaker 2 | 2:00 | Balance-error feature = the fingerprint of fraud; 4 models compared. |
| 5 Results | Speaker 3 | 2:00 | 1,637 of 1,643 frauds caught, **zero** false alarms. |
| 6 PR vs ROC | Speaker 3/4 | 1:30 | Same models: ROC 0.98 (looks great) but PR 0.59 — ROC lies on imbalanced data. |
| 7 Why it works | Speaker 4 | 1:30 | Our engineered feature drives 66% of decisions — it learned the real pattern. |
| 8 **LIVE DEMO** | Presenter | 2:00 | Legit → green; fraud → red; slide threshold → trade-off. |
| 9 Limitations | Speaker 4/5 | 1:30 | Synthetic-data caveat; ethics of auto-blocking. Earns credibility. |
| 10 Conclusion | All | 1:00 | Recap 0.998 PR-AUC; take questions. |

**Speaker notes are inside the PPTX** (View → Notes Page, or Presenter View) — each slide has a full script.

---

## The live-demo script (slide 8) — rehearse this

1. Click **"Load a typical LEGITIMATE transaction"** → **Score** → green ✅, low probability. Say: *"Ordinary cash-out, balances consistent — model clears it."*
2. Click **"Load the classic FRAUD pattern"** → **Score** → red 🚨, ~100%. Say: *"Large transfer draining the account to zero — the model catches it instantly."*
3. Drag the **threshold slider** down to ~0.20, re-score a borderline case. Say: *"Lower the threshold and we catch more fraud but raise false alarms — this is the cost decision a real bank must make."*
4. (Optional) Point at the engineered-features line under the inputs — *"you can watch the balance-error feature update live; when it's non-zero, the books don't add up."*

---

## Likely questions & strong answers

- **"Isn't 100% precision too good to be true?"** — Yes, and we say so in the report: PaySim's fraud follows a consistent rule our feature captures almost perfectly; real fraud adapts, so production numbers would be lower.
- **"Why not just use accuracy?"** — At 0.129% fraud, predicting 'never fraud' scores 99.87% accuracy while catching nothing. That's why we report Precision–Recall.
- **"Why did the tree models beat logistic regression so badly?"** — The fraud signal is non-linear — an interaction of balances and amount. A single linear boundary can't separate it; trees can.
- **"Why XGBoost in the app and not Random Forest?"** — They tied on accuracy (both 0.998 PR-AUC), but XGBoost trains 6.5× faster and saves to a few MB instead of hundreds — better for deployment.
- **"Could someone game this?"** — Yes; that's the adaptive-fraud limitation. Real systems retrain continually and add network/graph features.

---

## Final checklist before class

- [ ] `fraud_model.joblib` retrained on the **real** CSV (not demo)
- [ ] `streamlit run demo_app.py` tested on the actual presentation laptop
- [ ] Backup screen-recording of the demo saved locally
- [ ] Group name + member names on slide 1 (and the report title page)
- [ ] Every member has a speaking part
- [ ] Deck exported and the write-up PDF ready to upload after the talk
