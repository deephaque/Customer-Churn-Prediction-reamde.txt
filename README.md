# Customer Churn Prediction & Retention Copilot

Predicts which telecom customers will churn with a **Random Forest**, explains why (SHAP),
estimates business impact, and writes a short retention brief per customer.

## Dataset
IBM "Telco Customer Churn" sample dataset: 7,043 customers, 20 features, ~26.5% churn.
File: `data/telco_churn.csv`
Source: https://github.com/IBM/telco-customer-churn-on-icp4d (also on Kaggle as "Telco Customer Churn").

## Run
```
pip install -r requirements.txt
python churn.py
export ANTHROPIC_API_KEY=...   # optional: LLM-written briefs (template is used otherwise)
```

## Pipeline
1. **Clean:** coerce `TotalCharges` to numeric (blanks = new customers), encode `Churn` as 0/1.
2. **Model:** Random Forest (500 trees, `class_weight="balanced_subsample"`), compared with a Logistic Regression baseline.
3. **Evaluate:** ROC-AUC, PR-AUC (main metric, data is imbalanced), Brier score, calibration plot, 5-fold CV.
4. **Business impact:** profit when contacting the top-k% riskiest customers (editable cost / save-rate assumptions).
5. **Explain:** SHAP summary plot and per-customer top risk drivers.
6. **Copilot:** drivers map to a retention playbook; an LLM turns the facts into a brief, restricted to those facts only.

## Outputs
`outputs/model_comparison.csv`, `outputs/risk_scores.csv`, `outputs/calibration.png`, `outputs/shap_summary.png`
