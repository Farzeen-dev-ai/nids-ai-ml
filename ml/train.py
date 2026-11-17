"""
ml/train.py

Executes a 5-fold StratifiedKFold cross-validated hyperparameter search across 24 combinations
(120 fits total) using GridSearchCV on the training set (X_train, y_train) optimizing F1 score.
Serializes the refit best estimator to models/nids_pipeline.joblib and exports
data/ml_cv_results.csv and data/ml_cv_summary.json for evaluation handoff.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from ml.data_loader import load_and_split_data
from ml.pipeline import build_nids_pipeline

# 2 x 3 x 2 x 2 = 24 parameter combinations x 5 folds = 120 total fits
PARAM_GRID: Dict[str, Any] = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [12, 20, None],
    "classifier__min_samples_split": [2, 10],
    "classifier__class_weight": [None, "balanced"],
}


def run_hyperparameter_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> GridSearchCV:
    """
    Executes 5-fold StratifiedKFold GridSearchCV across PARAM_GRID,
    optimizing refit="f1" while tracking roc_auc, precision, and recall.
    """
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring={
            "f1": "f1",
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
        },
        refit="f1",
        cv=cv_splitter,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    print(f"[ml.train] Starting GridSearchCV across {len(grid_search.param_grid)} param combinations (120 fits)...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"[ml.train] Completed 120 fits in {duration:.2f} seconds ({duration / 60:.2f} min).")
    print(f"[ml.train] Best CV F1 Score: {grid_search.best_score_:.6f}")
    print(f"[ml.train] Best Hyperparameters: {grid_search.best_params_}")

    return grid_search


def main() -> None:
    # 1. Deterministic data partition
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()

    # 2. Pipeline construction
    pipeline = build_nids_pipeline(random_state=42)

    # 3. Stratified cross-validation hyperparameter search
    grid_search = run_hyperparameter_search(pipeline, X_train, y_train, random_state=42)

    # 4. Artifact destinations
    model_out = Path("models/nids_pipeline.joblib")
    cv_csv_out = Path("data/ml_cv_results.csv")
    cv_summary_out = Path("data/ml_cv_summary.json")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    cv_csv_out.parent.mkdir(parents=True, exist_ok=True)
    cv_summary_out.parent.mkdir(parents=True, exist_ok=True)

    # Save best serialized pipeline (gitignored)
    joblib.dump(grid_search.best_estimator_, model_out)
    print(f"[ml.train] Serialized best model to {model_out} ({model_out.stat().st_size / (1024 * 1024):.2f} MB)")

    # Save full 24-row CV results table sorted by rank_test_f1
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    if "rank_test_f1" in cv_results_df.columns:
        cv_results_df = cv_results_df.sort_values(by="rank_test_f1", ascending=True)
    cv_results_df.to_csv(cv_csv_out, index=False)
    print(f"[ml.train] Saved CV results table (24 rows) to {cv_csv_out}")

    # Save lightweight JSON summary handoff contract
    summary = {
        "best_cv_f1": round(float(grid_search.best_score_), 6),
        "best_params": grid_search.best_params_,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "num_features": int(len(feature_names)),
        "feature_names": feature_names,
    }
    with open(cv_summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ml.train] Saved CV summary handoff to {cv_summary_out}")


if __name__ == "__main__":
    main()
