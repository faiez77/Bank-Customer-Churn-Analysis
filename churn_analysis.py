"""
Bank Customer Churn Analysis — ML Pipeline
============================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

plt.rcParams['figure.dpi'] = 150

# =======================================================================
# 1. LOAD DATA

df = pd.read_csv("Bank Customer Churn Prediction.csv")
print("Shape:", df.shape)
print("Churn Rate:", df["churn"].mean())

df_original = df.copy()

# =======================================================================
# 2. FEATURE ENGINEERING

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

X = pd.get_dummies(X, columns=["country", "gender"], drop_first=True)

# =======================================================================
# 3. TRAIN/TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================================================================
# 4. LOGISTIC REGRESSION (baseline)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, y_prob_lr))

# =======================================================================
# 5. RANDOM FOREST (best model)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Threshold tuned to 0.6 (not the default 0.5) — trades a bit of recall
# for meaningfully better precision on the flagged high-risk list, so
# the retention team isn't chasing too many false positives.
threshold = 0.6
y_pred_rf = (y_prob_rf > threshold).astype(int)

print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, y_prob_rf))

results = df_original.loc[X_test.index].copy()
results["predicted_churn"] = y_pred_rf
results["churn_probability"] = y_prob_rf
results.to_csv("churn_predictions.csv", index=False)

# =======================================================================
# 6. TOP 50 HIGH-RISK CUSTOMERS

top_50 = results.sort_values(by="churn_probability", ascending=False).head(50)
top_50.to_csv("top_50_at_risk_customers.csv", index=False)

# =======================================================================
# 7. SHAP EXPLAINABILITY

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

shap_importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": np.abs(sv).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

shap_importance.to_csv("feature_importance.csv", index=False)
print("\n--- Top 5 features by mean |SHAP value| ---")
print(shap_importance.head(5).to_string(index=False))

# SHAP summary plot (bar) — for the Power BI dashboard / README
plt.figure(figsize=(7, 5))
shap.summary_plot(sv, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.close()
print("Saved: shap_feature_importance.png")

# =======================================================================
# 8. BUSINESS IMPACT

high_risk = results[results["churn_probability"] >= threshold]
balance_at_risk = high_risk["balance"].sum()
print(f"\nCustomers flagged high-risk (p >= {threshold}): {len(high_risk)}")
print(f"Total account balance at risk: {balance_at_risk:,.2f}")

print("\nFiles generated:")
print("1. churn_predictions.csv")
print("2. top_50_at_risk_customers.csv")
print("3. feature_importance.csv (SHAP-based, not Gini importance)")
print("4. shap_feature_importance.png")
