import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("output/summary.csv")

# Convert labels
df["label"] = df["class"].map({"AU":0, "TP":1})

# ---- Step 1: Create balanced subset only for threshold selection ----
min_count = df['label'].value_counts().min()
balanced_df = df.groupby('label').sample(min_count, random_state=42).reset_index(drop=True)

scores_bal = balanced_df["anomaly_score"].values
labels_bal = balanced_df["label"].values

# ---- Step 2: Find optimal threshold using balanced dataset ----
best_f1 = 0
best_t = 0

thresholds = np.linspace(0, scores_bal.max(), 200)

for t in thresholds:
    preds = (scores_bal >= t).astype(int)
    f1 = f1_score(labels_bal, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("Best Threshold (from balanced training selection):", best_t)
print("Best F1 on balanced data:", best_f1)

# ---- Step 3: Apply this threshold to FULL dataset ----
scores_full = df["anomaly_score"].values
labels_full = df["label"].values

final_preds = (scores_full >= best_t).astype(int)

print("\n--- Performance on Full Dataset ---")
print("Accuracy:", accuracy_score(labels_full, final_preds))
print("Precision:", precision_score(labels_full, final_preds))
print("Recall:", recall_score(labels_full, final_preds))
print("F1 Score:", f1_score(labels_full, final_preds))
print("ROC-AUC:", roc_auc_score(labels_full, scores_full))

# ---- Plot class distribution ----
tp = df[df["class"]=="TP"]["anomaly_score"]
au = df[df["class"]=="AU"]["anomaly_score"]

plt.hist(tp, bins=30, alpha=0.6, label="TP")
plt.hist(au, bins=30, alpha=0.6, label="AU")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.legend()
plt.title("TP vs AU Anomaly Score Distribution")
plt.show()

# ---- Confusion Matrix ----
cm = confusion_matrix(labels_full, final_preds)
print("\nConfusion Matrix (TN FP / FN TP):")
print(cm)
