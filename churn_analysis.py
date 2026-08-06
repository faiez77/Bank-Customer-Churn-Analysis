from google.colab import files
uploaded = files.upload()


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# 1. LOAD DATA 

df = pd.read_csv("Bank Customer Churn Prediction.csv")
print("Shape:", df.shape)
print("Churn Rate:", df["churn"].mean())

df_original = df.copy()


# 2. FEATURE ENGINEERING

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# Convert categorical → dummy
X = pd.get_dummies(X, columns=["country", "gender"], drop_first=True)


# 3. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. LOGISTIC REGRESSION

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


# 5. RANDOM FOREST (BEST MODEL)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

y_prob_rf = rf.predict_proba(X_test)[:, 1]

# 6. Threshold tuning (improves precision)
threshold = 0.6
y_pred_rf = (y_prob_rf > threshold).astype(int)

print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, y_prob_rf))


results = df_original.loc[X_test.index].copy()
results["predicted_churn"] = y_pred_rf
results["churn_probability"] = y_prob_rf

results.to_csv("churn_predictions_clean.csv", index=False)


# 7. TOP 50 HIGH RISK CUSTOMERS

top_50 = results.sort_values(by="churn_probability", ascending=False).head(50)
top_50.to_csv("top_50_at_risk_customers.csv", index=False)


# 8. FEATURE IMPORTANCE 

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance.to_csv("feature_importance.csv", index=False)

print("\n✅ Files Generated:")
print("1. churn_predictions_clean.csv")
print("2. top_50_at_risk_customers.csv")
print("3. feature_importance.csv")
