"""
University Student Mental Health - Early Warning Model
Trains Logistic Regression, Random Forest, and Decision Tree classifiers
to flag students at elevated depression risk, for use by a university
counseling center as a proactive (not diagnostic) screening aid.

Run: python train_model.py
Outputs: model.pkl (Decision Tree - chosen for interpretability),
         encoders.pkl, metrics.json, decision_tree.png
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, f1_score

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv("student_depression_dataset.csv")
print(f"Raw shape: {df.shape}")

# Clean stray quote characters present in some categorical values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.replace("'", "", regex=False).str.strip()

# Keep only students with a numeric Financial Stress value (a handful are '?')
df = df[pd.to_numeric(df["Financial Stress"], errors="coerce").notna()].copy()
df["Financial Stress"] = df["Financial Stress"].astype(float)

feature_cols = [
    "Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction",
    "Sleep Duration", "Dietary Habits",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress", "Family History of Mental Illness",
]
target_col = "Depression"

data = df[feature_cols + [target_col]].dropna()
print(f"Cleaned shape: {data.shape}")
print(data[target_col].value_counts())

# ---------------------------------------------------------------------
# 2. ENCODE
# ---------------------------------------------------------------------
cat_cols = ["Gender", "Sleep Duration", "Dietary Habits",
            "Have you ever had suicidal thoughts ?",
            "Family History of Mental Illness"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    data[c] = le.fit_transform(data[c])
    encoders[c] = le

X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# ---------------------------------------------------------------------
# 3. MODEL SHOWDOWN (matches the original README table)
# ---------------------------------------------------------------------
results = {}

print("\nLogistic Regression...")
logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
logreg.fit(X_train, y_train)
acc = accuracy_score(y_test, logreg.predict(X_test))
results["Logistic Regression"] = round(acc * 100, 1)
print(f"Accuracy: {acc:.3f}")

print("\nRandom Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
acc = accuracy_score(y_test, rf.predict(X_test))
results["Random Forest"] = round(acc * 100, 1)
print(f"Accuracy: {acc:.3f}")

print("\nDecision Tree (chosen model - interpretable)...")
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                             random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
acc = accuracy_score(y_test, dt_pred)
results["Decision Tree"] = round(acc * 100, 1)
print(f"Accuracy: {acc:.3f}")

report = classification_report(y_test, dt_pred, target_names=["No Depression", "Depression"],
                                output_dict=True)
print(classification_report(y_test, dt_pred, target_names=["No Depression", "Depression"]))

# ---------------------------------------------------------------------
# 4. FEATURE IMPORTANCE + TREE VISUAL (Decision Tree, the deployed model)
# ---------------------------------------------------------------------
importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(7, 4))
importances.plot(kind="barh", color="#6E4E9E")
plt.title("Top Risk Drivers - Decision Tree")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)

plt.figure(figsize=(18, 8))
plot_tree(dt, feature_names=feature_cols, class_names=["No Depression", "Depression"],
          filled=True, fontsize=7, max_depth=3)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
print("\nSaved feature_importance.png and decision_tree.png")

# ---------------------------------------------------------------------
# 5. SAVE ARTIFACTS (Decision Tree deployed — matches repo's chosen model)
# ---------------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(dt, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

metrics = {
    "model_showdown_accuracy_pct": results,
    "deployed_model": "Decision Tree",
    "deployed_model_classification_report": report,
    "n_records_used": int(len(data)),
    "feature_cols": feature_cols,
    "top_predictors": list(importances.sort_values(ascending=False).index[:5]),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print("\nSaved model.pkl, encoders.pkl, metrics.json")
print("\nDone.")
