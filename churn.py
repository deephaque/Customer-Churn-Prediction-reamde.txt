"""Churn prediction (Random Forest) + SHAP explainability + retention copilot.
Run: python churn.py"""
import os, warnings
import numpy as np, pandas as pd, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ---------- 1. Load & clean ----------
df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)  # blanks = brand-new customers
df["Churn"] = (df["Churn"] == "Yes").astype(int)

NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CAT = [c for c in df.columns if c not in NUM + ["customerID", "Churn"]]
X, y, ids = df[NUM + CAT], df["Churn"], df["customerID"]
print(f"Customers: {len(df)} | churn rate: {y.mean():.1%}")

X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
    X, y, ids, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
])

# ---------- 2. Models: Random Forest (main) vs Logistic Regression (baseline) ----------
rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
    n_estimators=500, min_samples_leaf=5, max_features="sqrt",
    class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

rows = []
for name, p in [("RandomForest (main)", rf), ("LogisticRegression (baseline)", lr)]:
    p.fit(X_tr, y_tr)
    pr = p.predict_proba(X_te)[:, 1]
    rows.append({"model": name, "ROC-AUC": roc_auc_score(y_te, pr),
                 "PR-AUC": average_precision_score(y_te, pr),
                 "Brier (lower=better)": brier_score_loss(y_te, pr)})
results = pd.DataFrame(rows).round(3)
print("\n=== Test-set comparison ===\n" + results.to_string(index=False))

cv = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                     scoring="average_precision", n_jobs=-1)
print(f"\nRandom Forest 5-fold CV PR-AUC: {cv.mean():.3f} +/- {cv.std():.3f}")

prob = rf.predict_proba(X_te)[:, 1]

# ---------- 3. Calibration ----------
frac_pos, mean_pred = calibration_curve(y_te, prob, n_bins=10)
plt.figure(figsize=(4.5, 4.5))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
plt.plot(mean_pred, frac_pos, "o-", label="Random Forest")
plt.xlabel("Predicted churn probability"); plt.ylabel("Actual churn rate")
plt.title("Calibration"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/calibration.png", dpi=150); plt.close()
print("Note: class_weight='balanced' inflates probabilities, so read them as risk scores / rankings.")

# ---------- 4. Business impact ----------
# Assumptions (edit): contacting costs money; a saved customer keeps paying ~12 months of their bill.
COST_PER_CONTACT, SAVE_RATE = 20, 0.30
value = X_te["MonthlyCharges"].values * 12
order = np.argsort(-prob)
print("\n=== Business impact: contact the top-k% riskiest customers ===")
best_k, best_profit = 0, -1e9
for k in [5, 10, 20, 30, 40]:
    top = order[: int(len(order) * k / 100)]
    caught = y_te.values[top]
    profit = (caught * value[top]).sum() * SAVE_RATE - len(top) * COST_PER_CONTACT
    print(f"top {k:>2}%: contacted={len(top):>4}, churners reached={caught.sum():>3}, net profit={profit:>9,.0f}")
    if profit > best_profit: best_k, best_profit = k, profit
print(f"-> Best targeting depth: top {best_k}% (profit {best_profit:,.0f})")

# ---------- 5. SHAP ----------
Xt_te = pd.DataFrame(rf.named_steps["pre"].transform(X_te),
                     columns=rf.named_steps["pre"].get_feature_names_out())
sv = shap.TreeExplainer(rf.named_steps["clf"]).shap_values(Xt_te)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
shap.summary_plot(sv, Xt_te, max_display=12, show=False)
plt.tight_layout(); plt.savefig("outputs/shap_summary.png", dpi=150); plt.close()

# ---------- 6. Retention copilot ----------
PLAYBOOK = {  # editable: original feature -> retention action
    "Contract": "offer a discounted 1-year or 2-year plan to move off month-to-month",
    "tenure": "send an onboarding check-in and early-life perks",
    "MonthlyCharges": "review the plan and offer a cheaper bundle or loyalty discount",
    "TotalCharges": "offer a loyalty reward for long-term spend",
    "InternetService": "offer a service-quality check or a speed upgrade",
    "OnlineSecurity": "offer a free trial of online security",
    "TechSupport": "offer free tech support for 3 months",
    "OnlineBackup": "offer a free backup add-on",
    "DeviceProtection": "offer a discounted device protection plan",
    "PaymentMethod": "nudge toward auto-pay with a small discount",
    "PaperlessBilling": "send a billing clarity email and payment review",
    "StreamingTV": "offer a streaming bundle deal",
    "StreamingMovies": "offer a streaming bundle deal",
    "MultipleLines": "offer a family-lines discount",
    "SeniorCitizen": "offer a senior-friendly plan and dedicated support line",
}

def base_feature(col):
    for c in NUM:
        if col == f"num__{c}": return c
    return next(c for c in CAT if col.startswith(f"cat__{c}_"))

def top_drivers(i, n=3):
    idx = np.argsort(-np.abs(sv[i]))[:n]
    return [(base_feature(Xt_te.columns[j]), Xt_te.columns[j].split("__", 1)[1], sv[i][j]) for j in idx]

def retention_summary(i):
    """Grounded brief. Uses an LLM if ANTHROPIC_API_KEY is set, else a template."""
    drivers = top_drivers(i)
    risk_drivers = [(b, f) for b, f, s in drivers if s > 0]
    actions = list(dict.fromkeys(PLAYBOOK[b] for b, _ in risk_drivers))[:2] or ["no urgent action needed"]
    facts = (f"Customer {id_te.iloc[i]}: churn risk score {prob[i]:.0%}. Profile: {X_te.iloc[i].to_dict()}. "
             "Top risk drivers: " + "; ".join(f for _, f in risk_drivers))
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5-5", max_tokens=250,
            messages=[{"role": "user", "content":
                "Write a 3-sentence retention brief for a customer success manager. Use ONLY the facts "
                f"below; do not invent data.\nFACTS: {facts}\nALLOWED ACTIONS: {actions}"}])
        return msg.content[0].text
    return (f"Customer {id_te.iloc[i]} has a {prob[i]:.0%} churn risk score, mainly driven by "
            + ", ".join(f for _, f in risk_drivers) + ". Recommended: " + "; ".join(actions) + ".")

print("\n=== Retention copilot: 3 highest-risk customers ===")
for i in order[:3]:
    print("\n" + retention_summary(i))

results.to_csv("outputs/model_comparison.csv", index=False)
pd.DataFrame({"customerID": id_te.values, "churn_risk": prob.round(3), "actual_churn": y_te.values}) \
  .sort_values("churn_risk", ascending=False).to_csv("outputs/risk_scores.csv", index=False)
