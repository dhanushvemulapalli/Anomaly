import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import time
import matplotlib.pyplot as plt


parent_start = time.time()
# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("../../output/summary.csv")

# Convert class labels TP/AU → binary
df["label"] = df["class"].map({"TP": 1, "AU": 0})

# Drop non-feature columns
X = df.drop(columns=["image", "class", "label"])
y = df["label"]


# ---------------------------
# Train / Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


# ---------------------------
# Scaling for SVM
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("/n")
print("===============================================================================")
print("/n")
# ---------------------------
# Train SVM (RBF kernel)
# ---------------------------
print("\n🔹 Training SVM...")
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)


# ---------------------------
# Train Random Forest
# ---------------------------
print("🔹 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

print("/n")
print("===============================================================================")
print("/n")

# ---------------------------
# Evaluation Helper
# ---------------------------
def evaluate(model, X_test, y_test, model_name):
    print(f"\n🔹 Evaluating {model_name}...")
    start = time.time()
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:,1]
        if hasattr(model, "predict_proba") else None
    )
    
    print(f"\n📌 Results for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    
    if y_prob is not None:
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(f"Evaluation Time: {end - start:.2f} seconds")
    print("\n")


# ---------------------------
# Evaluate Both Models
# ---------------------------
evaluate(svm_model, X_test_scaled, y_test, "SVM (RBF)")
evaluate(rf_model, X_test, y_test, "Random Forest")

parent_end = time.time()
print(f"\n⏱️ Total time taken for training and evaluation: {parent_end - parent_start:.2f} seconds")

print("\n🎉 Training complete!")

print("/n")
print("===============================================================================")
print("/n")


print("Starting Hyperparameter Tuning...")


# from sklearn.model_selection import GridSearchCV

# # ---------------------
# # SVM tuning
# # ---------------------
# svm_params = {
#     "C": [0.1, 1, 10, 50],
#     "gamma": ["scale", 0.01, 0.001],
#     "kernel": ["rbf"]
# }

# svm_grid = GridSearchCV(
#     estimator=SVC(probability=True),
#     param_grid=svm_params,
#     scoring="f1",
#     cv=3,
#     n_jobs=-1,
#     verbose=1
# )

# svm_grid.fit(X_train_scaled, y_train)

# print("\n🔥 Best SVM Params:", svm_grid.best_params_)
# print("Best SVM F1 Score:", svm_grid.best_score_)


# # ---------------------
# # Random Forest tuning
# # ---------------------
# rf_params = {
#     "n_estimators": [200, 300, 500],
#     "max_depth": [None, 20, 30, 50],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4]
# }

# rf_grid = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=rf_params,
#     scoring="f1",
#     cv=3,
#     n_jobs=-1,
#     verbose=1
# )

# rf_grid.fit(X_train, y_train)

# print("\n🔥 Best RF Params:", rf_grid.best_params_)
# print("Best RF F1 Score:", rf_grid.best_score_)

# print("/n")
# print("===============================================================================")
# print("/n")


print("\n🎉 Training Ensemble Model...")

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

voting = VotingClassifier(
    estimators=[
        ("svm", svm_grid.best_estimator_ if 'svm_grid' in globals() else svm_model),
        ("rf", rf_grid.best_estimator_ if 'rf_grid' in globals() else rf_model),
        ("lr", LogisticRegression(max_iter=2000))
    ],
    voting="soft"
)

voting.fit(X_train_scaled, y_train)

y_pred = voting.predict(X_test_scaled)

print("\n🧠 Ensemble Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("/n")
print("===============================================================================")
print("/n")

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

probs = voting.predict_proba(X_test_scaled)[:,1]

print("\n🔧 Threshold Tuning\nThreshold | Precision | Recall | F1")
best_f1 = 0
best_thresh = 0.5

for thresh in np.arange(0.1, 0.9, 0.05):
    preds = (probs >= thresh).astype(int)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    
    print(f"{thresh:.2f}      | {precision:.3f}     | {recall:.3f} | {f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\n🏆 Best Threshold: {best_thresh}, Best F1: {best_f1}")


import seaborn as sns
from sklearn.metrics import RocCurveDisplay, precision_recall_curve

# =============================== #
#        VISUALIZATION SECTION    #
# =============================== #

# ---- 1. Feature Importance (Random Forest) ----
plt.figure(figsize=(10,5))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=45, ha="right")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()


# ---- 2. ROC Curve ----
plt.figure(figsize=(6,6))
RocCurveDisplay.from_estimator(voting, X_test_scaled, y_test)
plt.title("ROC Curve - Ensemble Model")
plt.savefig("roc_curve.png", dpi=300)
plt.show()


# ---- 3. Precision-Recall Curve (compatible version) ----
precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve - Ensemble Model")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("pr_curve.png", dpi=300)
plt.show()


# ---- 4. Confusion Matrix Heatmap ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AU", "TP"], yticklabels=["AU", "TP"])
plt.title("Confusion Matrix - Ensemble Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ---- 5. Threshold vs F1 Plot ----
thresholds = np.arange(0.1, 0.9, 0.05)
f1_values = []

for thresh in thresholds:
    preds = (probs >= thresh).astype(int)
    f1_values.append(f1_score(y_test, preds))

plt.figure(figsize=(8,5))
plt.plot(thresholds, f1_values, marker="o")
plt.axvline(best_thresh, linestyle="--", color="red", label=f"Best Threshold={best_thresh:.2f}")
plt.title("Threshold vs F1 Score")
plt.xlabel("Decision Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.savefig("threshold_vs_f1.png", dpi=300)
plt.show()

print("\n📊 Graphs generated and saved successfully.")
