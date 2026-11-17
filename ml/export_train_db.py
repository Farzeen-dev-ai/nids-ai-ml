"""
ml/export_train_db.py

Prepares a dedicated training SQLite database (data/nids_train.db) from the full database (data/nids_full.db).
Drops non-training metadata and EDA summary tables, vacuums outside open transactions,
and enforces strict try/finally resource closure to prevent Windows 11 file locks.
"""

import os
import sqlite3
from pathlib import Path
from typing import List

NON_TRAINING_TABLES: List[str] = [
    "etl_offsets",
    "eda_class_distribution",
    "eda_summary_kpis",
    "eda_metadata",
    "eda_correlation_matrix",
    "eda_scatter_sample",
]


def export_train_db(
    source_db: Path = Path("data/nids_full.db"),
    target_db: Path = Path("data/nids_train.db"),
) -> int:
    """
    Copies data/nids_full.db to data/nids_train.db, drops EDA/offset tables,
    runs VACUUM outside open transactions, and verifies the row count.
    """
    if not source_db.exists():
        raise FileNotFoundError(f"Source database not found: {source_db.resolve()}")

    target_db.parent.mkdir(parents=True, exist_ok=True)
    if target_db.exists():
        target_db.unlink()

    # Step 1: Online SQLite backup to destination file
    src_conn = sqlite3.connect(str(source_db))
    dst_conn = sqlite3.connect(str(target_db))
    try:
        src_conn.backup(dst_conn)
    finally:
        src_conn.close()
        dst_conn.close()

    # Step 2: Drop non-training tables and vacuum the standalone database
    clean_conn = sqlite3.connect(str(target_db))
    clean_conn.isolation_level = None  # Autocommit mode: ensures VACUUM runs outside active transaction
    try:
        cursor = clean_conn.cursor()
        for tbl in NON_TRAINING_TABLES:
            cursor.execute(f"DROP TABLE IF EXISTS {tbl};")
        cursor.execute("VACUUM;")
        cursor.execute("SELECT COUNT(*) FROM network_flows;")
        row = cursor.fetchone()
        row_count = int(row[0]) if row else 0
    finally:
        clean_conn.close()

    size_mb = os.path.getsize(target_db) / (1024 * 1024)
    print(f"[export_train_db] Successfully exported {row_count:,} rows to {target_db} ({size_mb:.2f} MB)")
    return row_count


if __name__ == "__main__":
    export_train_db()
