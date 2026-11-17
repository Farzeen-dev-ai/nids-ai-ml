"""
ml/pipeline.py

Builds a unified scikit-learn Pipeline encapsulating preprocessing and estimator
to strictly prevent data leakage across cross-validation folds.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_nids_pipeline(random_state: int = 42) -> Pipeline:
    """
    Constructs a unified, atomic scikit-learn Pipeline with:
    1. SimpleImputer (median strategy, keep_empty_features=True)
    2. StandardScaler (z-score normalization)
    3. RandomForestClassifier (n_jobs=1 to allow outer multiprocessing)
    """
    nids_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=1,  # Outer GridSearchCV handles multiprocessing via n_jobs=-1
                ),
            ),
        ]
    )
    return nids_pipeline


if __name__ == "__main__":
    pipeline = build_nids_pipeline()
    print("[ml.pipeline] Successfully constructed pipeline:")
    for step_name, step_obj in pipeline.steps:
        print(f"  - Step: {step_name} -> {step_obj}")
