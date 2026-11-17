"""
ml/data_loader.py

Loads network flows from SQLite, applies leakage-prevention column exclusions,
extracts numeric features and binary target, and executes a stratified 80/20 train/test split.
"""

import sqlite3
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split

DROP_COLUMNS: List[str] = ["flow_id", "label", "is_attack", "fwd_header_length_1"]


def load_and_split_data(
    db_path: str = "data/nids_train.db",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Reads network_flows from db_path (falling back to data/nids_full.db if missing),
    drops non-feature / target-leaking columns, extracts numeric features and binary target,
    and returns a stratified train/test split.
    """
    primary = Path(db_path)
    fallback = Path("data/nids_full.db")
    resolved_path = primary if primary.exists() else fallback

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Neither primary ({primary}) nor fallback ({fallback}) database exists. "
            "Execute `python -m ml.export_train_db` first."
        )

    conn = sqlite3.connect(str(resolved_path))
    try:
        df = pd.read_sql_query("SELECT * FROM network_flows", conn)
    finally:
        conn.close()

    if "is_attack" not in df.columns:
        raise KeyError("Expected 'is_attack' column in network_flows table.")

    # Target vector: binary indicator (0 for BENIGN, 1 for ATTACK)
    y = df["is_attack"].astype(int).copy()

    # Drop non-feature, leakage, and duplicate columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    feature_df = df.drop(columns=cols_to_drop)

    # Discovered numeric features
    X = feature_df.select_dtypes(include=["number"]).copy()

    # Stratified 80/20 train/test partition before any preprocessing or fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    feature_names = list(X.columns)

    print(
        f"[data_loader] Loaded {len(df):,} total rows from {resolved_path} | "
        f"Features: {X.shape[1]} | Train: {len(y_train):,} | Test: {len(y_test):,}"
    )

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_split_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape} (Attack: {y_train.sum():,}, Benign: {(y_train == 0).sum():,})")
    print(f"y_test shape:  {y_test.shape} (Attack: {y_test.sum():,}, Benign: {(y_test == 0).sum():,})")
