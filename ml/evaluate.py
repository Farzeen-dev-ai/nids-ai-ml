"""
ml/evaluate.py

Decoupled held-out test evaluation module.
Loads the serialized models/nids_pipeline.joblib and data/ml_cv_summary.json,
reconstructs the exact 5,684-row sequestered held-out test set,
evaluates performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC),
extracts downsampled ROC curve points and top-20 Gini feature importances,
and writes data/ml_evaluation_report.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from ml.data_loader import load_and_split_data

SAMPLING_LIMITATION_NOTE: str = (
    "Methodological Note on Contiguous Batch Sampling: "
    "Phase 1 extracted 300 evenly spaced offsets of 100 contiguous rows each across the 2.83M-row dataset. "
    "Because adjacent packets within a 100-row contiguous batch can belong to the same network session "
    "or temporal burst, row-level stratified splitting can place temporally adjacent flows across both "
    "train and test partitions, which may inflate tree-ensemble test metrics."
)


def evaluate_and_serialize(
    best_pipeline: Pipeline,
    cv_summary: Dict[str, Any],
    X_test,
    y_test,
    n_train: int,
    feature_names: List[str],
    report_out: Path = Path("data/ml_evaluation_report.json"),
) -> Dict[str, Any]:
    """
    Evaluates the trained pipeline on the sequestered test set and serializes the report to JSON.
    """
    n_test = int(len(y_test))

    # Single atomic pipeline inference
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    # Classification Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))

    # Confusion Matrix: [tn, fp, fn, tp]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm_raw = [int(tn), int(fp), int(fn), int(tp)]

    # Downsampled ROC Curve (maximum 100 points for lightweight frontend rendering)
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
    sample_indices = np.linspace(0, len(fpr_arr) - 1, min(len(fpr_arr), 100), dtype=int)
    fpr_sampled = [round(float(fpr_arr[i]), 6) for i in sample_indices]
    tpr_sampled = [round(float(tpr_arr[i]), 6) for i in sample_indices]

    # Top-20 Gini Feature Importances with strict feature count alignment check
    rf_clf = best_pipeline.named_steps["classifier"]
    importances = rf_clf.feature_importances_
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Feature alignment mismatch: {len(feature_names)} feature names provided "
            f"vs {len(importances)} classifier feature importances."
        )

    paired_importances = sorted(
        zip(feature_names, importances),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )[:20]

    top_20_features = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in paired_importances
    ]

    report: Dict[str, Any] = {
        "model_name": "RandomForestClassifier",
        "best_cv_f1": cv_summary.get("best_cv_f1"),
        "best_params": cv_summary.get("best_params"),
        "n_train": n_train,
        "n_test": n_test,
        "num_features": len(feature_names),
        "metrics": {
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "roc_auc": round(roc_auc, 6),
            "pr_auc": round(pr_auc, 6),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "raw": cm_raw,
        },
        "roc_curve": {
            "fpr": fpr_sampled,
            "tpr": tpr_sampled,
        },
        "top_features": top_20_features,
        "methodological_note": SAMPLING_LIMITATION_NOTE,
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[ml.evaluate] Wrote evaluation report to {report_out}")
    print(f"  Accuracy:  {accuracy:.4%}")
    print(f"  Precision: {precision:.4%}")
    print(f"  Recall:    {recall:.4%}")
    print(f"  F1-Score:  {f1:.4%}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    return report


def main() -> None:
    model_path = Path("models/nids_pipeline.joblib")
    summary_path = Path("data/ml_cv_summary.json")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {model_path}. "
            "Execute `python -m ml.train` first."
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"CV summary artifact not found at {summary_path}. "
            "Execute `python -m ml.train` first."
        )

    best_pipeline = joblib.load(model_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        cv_summary = json.load(f)

    # Reconstruct exact deterministic partition
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()
    n_train = len(y_train)

    evaluate_and_serialize(
        best_pipeline=best_pipeline,
        cv_summary=cv_summary,
        X_test=X_test,
        y_test=y_test,
        n_train=n_train,
        feature_names=feature_names,
    )


if __name__ == "__main__":
    main()
