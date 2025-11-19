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
