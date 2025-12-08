import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ==========================
# CONFIG
# ==========================

DATA_PATH = Path("../../output/summary.csv")
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_RF_SEARCH_ITER = 40  # for tuned RF (RandomizedSearchCV)


# ==========================
# HELPERS
# ==========================

def print_separator():
    print("\n" + "=" * 80 + "\n")


def evaluate_default_threshold(y_test, y_prob):
    """Evaluate using default threshold 0.5."""
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return y_pred, acc, prec, rec, f1


def tune_threshold(y_test, y_prob, label="Model"):
    """Search threshold in [0.1, 0.9) step 0.05 to maximize F1."""
    print(f"\n🔧 Threshold Tuning for {label}")
    print("Threshold | Precision | Recall | F1")
    best_f1 = 0.0
    best_thresh = 0.5

    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_values = []

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)

        # Avoid degenerate all-0 or all-1 cases if they occur
        if preds.sum() == 0 or preds.sum() == len(preds):
            f1_values.append(0.0)
            continue

        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        f1_values.append(f1)

        print(f"{thresh:.2f}      | {prec:.3f}     | {rec:.3f} | {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n🏆 Best Threshold for {label}: {best_thresh:.2f}, Best F1: {best_f1:.4f}")
    return best_thresh, best_f1, thresholds, f1_values


def train_and_eval_model(name, model, X_train, y_train, X_test, y_test):
    """Fit model, evaluate at 0.5 and best threshold; return summary."""
    print_separator()
    print(f"🔹 Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Get probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision_function -> convert to [0,1] via sigmoid-like scaling
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    # Default threshold 0.5
    print(f"\n📌 {name} @ threshold 0.5")
    y_pred_default, acc, prec, rec, f1 = evaluate_default_threshold(y_test, y_prob)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    try:
        roc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC:   {roc:.4f}")
    except Exception:
        roc = None

    print("\nConfusion Matrix (0.5):")
    print(confusion_matrix(y_test, y_pred_default))

    print("\nClassification Report (0.5):")
    print(classification_report(y_test, y_pred_default, zero_division=0))

    # Threshold tuning for F1
    best_thresh, best_f1, thresholds, f1_values = tune_threshold(y_test, y_prob, label=name)

    y_pred_best = (y_prob >= best_thresh).astype(int)
    acc_best = accuracy_score(y_test, y_pred_best)
    prec_best = precision_score(y_test, y_pred_best, zero_division=0)
    rec_best = recall_score(y_test, y_pred_best, zero_division=0)

    print(f"\n📌 {name} @ BEST threshold {best_thresh:.2f}")
    print(f"Accuracy:  {acc_best:.4f}")
    print(f"Precision: {prec_best:.4f}")
    print(f"Recall:    {rec_best:.4f}")
    print(f"F1 Score:  {best_f1:.4f}")
    print("Confusion Matrix (best):")
    print(confusion_matrix(y_test, y_pred_best))

    total_time = time.time() - start

    summary = {
        "name": name,
        "model": model,
        "y_prob": y_prob,
        "best_thresh": best_thresh,
        "best_f1": best_f1,
        "acc_best": acc_best,
        "prec_best": prec_best,
        "rec_best": rec_best,
        "roc_auc": roc,
        "train_eval_time": total_time,
        "thresholds": thresholds,
        "f1_curve": f1_values,
    }
    return summary


# ==========================
# MAIN
# ==========================

def main():
    global RANDOM_STATE

    parent_start = time.time()

    # ---------------------------
    # Load dataset
    # ---------------------------
    df = pd.read_csv(DATA_PATH)

    # Map TP/AU → 1/0
    df["label"] = df["class"].map({"TP": 1, "AU": 0})

    print("Number of TP samples:", df[df["class"] == "TP"].shape[0])
    print("Number of AU samples:", df[df["class"] == "AU"].shape[0])

    # Features / labels
    X = df.drop(columns=["image", "class", "label"])
    y = df["label"]

    # ---------------------------
    # Train-test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # For LR in stacking (and if you want later SVM etc.)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_summaries = []

    # ============================================================
    # 1. Baseline Random Forest
    # ============================================================
    rf_baseline = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    summary_rf = train_and_eval_model(
        "RandomForest (baseline)",
        rf_baseline,
        X_train, y_train, X_test, y_test
    )
    model_summaries.append(summary_rf)

    # ============================================================
    # 2. Tuned Random Forest (using fixed best params, no CV)
    # ============================================================
    print_separator()
    print("🔹 Training Random Forest (tuned, fixed hyperparameters)...")

    # Hard-coded best params from previous RandomizedSearchCV run
    best_rf_params = {
        "bootstrap": True,
        "max_depth": 20,
        "max_features": "log2",
        "min_samples_leaf": 1,
        "min_samples_split": 5,
        "n_estimators": 249,
    }

    rf_tuned = RandomForestClassifier(
        **best_rf_params,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    summary_rf_tuned = train_and_eval_model(
        "RandomForest (tuned)",
        rf_tuned,
        X_train, y_train, X_test, y_test
    )
    model_summaries.append(summary_rf_tuned)

    # ============================================================
    # 3. XGBoost
    # ============================================================
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    summary_xgb = train_and_eval_model(
        "XGBoost",
        xgb_model,
        X_train, y_train, X_test, y_test
    )
    model_summaries.append(summary_xgb)

    # ============================================================
    # 4. LightGBM
    # ============================================================
    lgbm_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    summary_lgbm = train_and_eval_model(
        "LightGBM",
        lgbm_model,
        X_train, y_train, X_test, y_test
    )
    model_summaries.append(summary_lgbm)

    # ============================================================
    # 5. CatBoost
    # ============================================================
    cat_model = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=800,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=RANDOM_STATE,
        verbose=100
    )
    summary_cat = train_and_eval_model(
        "CatBoost",
        cat_model,
        X_train, y_train, X_test, y_test
    )
    model_summaries.append(summary_cat)

    # ============================================================
    # 6. Stacking (CatBoost + tuned RF + LR)
    # ============================================================
    # Recreate fresh base estimators (StackingClassifier will clone & fit them)
    cb_base = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=800,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=0
    )
    rf_base_tuned = RandomForestClassifier(
        **best_rf_params,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    lr_base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

    stacking = StackingClassifier(
        estimators=[
            ("cb", cb_base),
            ("rf", rf_base_tuned),
            ("lr", lr_base),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        passthrough=True,    # also pass original features to final estimator
        n_jobs=-1
    )

    summary_stack = train_and_eval_model(
        "Stacking (CatBoost + RF_tuned + LR)",
        stacking,
        X_train, y_train, X_test, y_test
    )
    model_summaries.append(summary_stack)

    print_separator()
    print("🏁 All models trained & evaluated.\n")

    # ============================================================
    # Compare models by best F1
    # ============================================================
    print("📊 Model comparison (sorted by best F1):\n")
    model_summaries_sorted = sorted(model_summaries, key=lambda d: d["best_f1"], reverse=True)
    for s in model_summaries_sorted:
        print(f"{s['name']:<35} | Best F1: {s['best_f1']:.4f} | "
              f"Precision: {s['prec_best']:.4f} | Recall: {s['rec_best']:.4f}")

    best_model = model_summaries_sorted[0]
    print_separator()
    print(f"🏆 BEST MODEL (by F1): {best_model['name']}")
    print(f"   Best Threshold: {best_model['best_thresh']:.2f}")
    print(f"   F1: {best_model['best_f1']:.4f}")
    print(f"   Precision: {best_model['prec_best']:.4f}")
    print(f"   Recall: {best_model['rec_best']:.4f}")
    print(f"   ROC-AUC: {best_model['roc_auc']:.4f}" if best_model['roc_auc'] is not None else "")

    parent_end = time.time()
    print(f"\n⏱️ Total time for entire pipeline: {parent_end - parent_start:.2f} seconds")

    # ============================================================
    # OPTIONAL: Plot threshold vs F1 for best model
    # ============================================================
    thresholds = best_model["thresholds"]
    f1_curve = best_model["f1_curve"]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_curve, marker="o")
    plt.axvline(best_model["best_thresh"], linestyle="--", color="red",
                label=f"Best Threshold = {best_model['best_thresh']:.2f}")
    plt.title(f"Threshold vs F1 Score - {best_model['name']}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("best_model_threshold_vs_f1.png", dpi=300)
    plt.show()

    # Confusion matrix heatmap for best model at best threshold
    y_pred_best = (best_model["y_prob"] >= best_model["best_thresh"]).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["AU", "TP"], yticklabels=["AU", "TP"])
    plt.title(f"Confusion Matrix - {best_model['name']} (Best Threshold)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("best_model_confusion_matrix.png", dpi=300)
    plt.show()

    print("\n📊 Saved:")
    print(" - best_model_threshold_vs_f1.png")
    print(" - best_model_confusion_matrix.png")


    # ============================================================
    # Plots: Precision–Recall, ROC, Feature Importance for best model
    # ============================================================
    from sklearn.metrics import (
        RocCurveDisplay,
        PrecisionRecallDisplay,
        precision_recall_curve,
    )

    saved_files = []

    # Best-threshold predictions (you already computed best_thresh earlier)
    y_pred_best = (best_model["y_prob"] >= best_model["best_thresh"]).astype(int)

    # ---------- Precision–Recall Curve ----------
    precision, recall, thresholds = precision_recall_curve(y_test, best_model["y_prob"])

    plt.figure(figsize=(8, 5))
    PrecisionRecallDisplay.from_predictions(
        y_test,
        best_model["y_prob"],
        name=best_model["name"],
    )
    plt.title(f"Precision–Recall Curve – {best_model['name']}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("best_model_precision_recall_curve.png", dpi=300)
    plt.show()
    saved_files.append("best_model_precision_recall_curve.png")

    # ---------- ROC Curve ----------
    plt.figure(figsize=(8, 5))
    RocCurveDisplay.from_estimator(
        best_model["model"],
        X_test,
        y_test,
        name=best_model["name"],
    )
    plt.title(f"ROC Curve – {best_model['name']}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("best_model_roc_curve.png", dpi=300)
    plt.show()
    saved_files.append("best_model_roc_curve.png")

    # ---------- Feature Importance (if available) ----------
    if hasattr(best_model["model"], "feature_importances_"):
        importances = best_model["model"].feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance – {best_model['name']}")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), X.columns[indices], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("best_model_feature_importance.png", dpi=300)
        plt.show()
        saved_files.append("best_model_feature_importance.png")
    else:
        print(f"\n⚠️ Feature importance not available for {best_model['name']}.")

    print("\n📊 Saved:")
    for fname in saved_files:
        print(f" - {fname}")

if __name__ == "__main__":
    main()
