from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATA_PROCESSED

try:
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover
    duckdb = None


DASHBOARD_DIR = DATA_PROCESSED / "dashboard"
CACHE_DIR = DASHBOARD_DIR / "cache"
DB_PATH = DASHBOARD_DIR / "dashboard.duckdb"


class DashboardStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.dashboard_dir = self.db_path.parent
        self.cache_dir = CACHE_DIR
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = "duckdb" if duckdb is not None else "sqlite"
        self.initialize()

    def connect(self):
        if duckdb is not None:
            return duckdb.connect(str(self.db_path))
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _read_frame(
        self,
        conn: Any,
        query: str,
        params: list[Any] | tuple[Any, ...] | None = None,
    ) -> pd.DataFrame:
        if self.backend == "duckdb":
            if params:
                return conn.execute(query, params).df()
            return conn.execute(query).df()
        return pd.read_sql_query(query, conn, params=params)

    def initialize(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS review_fetch_jobs (
                job_id TEXT PRIMARY KEY,
                job_key TEXT UNIQUE,
                app_id TEXT,
                app_name TEXT,
                date_from TEXT,
                date_to TEXT,
                review_limit INTEGER,
                fetched_at TEXT,
                model_id TEXT,
                status TEXT,
                review_cache_path TEXT,
                prediction_cache_path TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reviews_fact (
                review_id_ext TEXT,
                source_job_id TEXT,
                app_id TEXT,
                app_name TEXT,
                review_date TEXT,
                rating REAL,
                review_text_raw TEXT,
                review_text_clean TEXT,
                PRIMARY KEY (review_id_ext, source_job_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS predictions_fact (
                review_id_ext TEXT,
                source_job_id TEXT,
                model_id TEXT,
                aspect TEXT,
                pred_label TEXT,
                confidence REAL,
                prob_negative REAL,
                prob_neutral REAL,
                prob_positive REAL,
                PRIMARY KEY (review_id_ext, source_job_id, model_id, aspect)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS gold_subset_fact (
                item_id TEXT PRIMARY KEY,
                review_id TEXT,
                aspect TEXT,
                review_text TEXT,
                label TEXT,
                aspect_present INTEGER,
                confidence INTEGER,
                notes TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS gold_eval_fact (
                model_id TEXT,
                item_id TEXT,
                pred_label TEXT,
                pred_confidence REAL,
                sentiment_match INTEGER,
                PRIMARY KEY (model_id, item_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS experiment_registry (
                model_id TEXT PRIMARY KEY,
                display_name TEXT,
                family TEXT,
                epoch INTEGER,
                training_regime TEXT,
                model_type TEXT,
                source_path TEXT,
                rank_weak_label INTEGER,
                rank_gold_subset INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_fetch_audit_fact (
                source_job_id TEXT,
                app_id TEXT,
                app_name TEXT,
                stage_order INTEGER,
                stage_name TEXT,
                count INTEGER,
                PRIMARY KEY (source_job_id, app_id, stage_order)
            )
            """,
        ]

        with self.connect() as conn:
            for statement in statements:
                conn.execute(statement)
            conn.commit()

    def job_cache_paths(self, job_id: str) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"{job_id}_reviews.csv",
            self.cache_dir / f"{job_id}_predictions.csv",
        )

    def find_cached_job(self, job_key: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            df = self._read_frame(
                conn,
                "SELECT * FROM review_fetch_jobs WHERE job_key = ? AND status = 'completed' LIMIT 1",
                params=[job_key],
            )
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def save_live_job(
        self,
        job_meta: dict[str, Any],
        reviews_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        fetch_audit_df: pd.DataFrame | None = None,
    ) -> None:
        review_cache_path, prediction_cache_path = self.job_cache_paths(job_meta["job_id"])
        reviews_df.to_csv(review_cache_path, index=False, encoding="utf-8")
        predictions_df.to_csv(prediction_cache_path, index=False, encoding="utf-8")

        payload = {
            **job_meta,
            "review_cache_path": str(review_cache_path),
            "prediction_cache_path": str(prediction_cache_path),
        }

        review_limit_raw = payload.get("review_limit")
        if pd.isna(review_limit_raw):
            review_limit_db = None
        elif review_limit_raw in (None, "", "None"):
            review_limit_db = None
        else:
            try:
                review_limit_db = int(float(review_limit_raw))
            except (TypeError, ValueError):
                review_limit_db = None

        review_rows = [
            (
                str(row.review_id_ext),
                payload["job_id"],
                row.app_id,
                row.app_name,
                row.review_date,
                float(row.rating) if pd.notna(row.rating) else None,
                row.review_text_raw,
                row.review_text_clean,
            )
            for row in reviews_df.itertuples(index=False)
        ]
        pred_rows = [
            (
                str(row.review_id_ext),
                payload["job_id"],
                payload["model_id"],
                row.aspect,
                row.pred_label,
                float(row.confidence),
                float(row.prob_negative),
                float(row.prob_neutral),
                float(row.prob_positive),
            )
            for row in predictions_df.itertuples(index=False)
        ]

        with self.connect() as conn:
            conn.execute("DELETE FROM review_fetch_jobs WHERE job_id = ?", [payload["job_id"]])
            conn.execute("DELETE FROM reviews_fact WHERE source_job_id = ?", [payload["job_id"]])
            conn.execute("DELETE FROM predictions_fact WHERE source_job_id = ?", [payload["job_id"]])
            conn.execute("DELETE FROM live_fetch_audit_fact WHERE source_job_id = ?", [payload["job_id"]])
            conn.execute(
                """
                INSERT INTO review_fetch_jobs (
                    job_id, job_key, app_id, app_name, date_from, date_to, review_limit,
                    fetched_at, model_id, status, review_cache_path, prediction_cache_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    payload["job_id"],
                    payload["job_key"],
                    payload["app_id"],
                    payload["app_name"],
                    payload["date_from"],
                    payload["date_to"],
                    review_limit_db,
                    payload["fetched_at"],
                    payload["model_id"],
                    payload["status"],
                    payload["review_cache_path"],
                    payload["prediction_cache_path"],
                ],
            )
            conn.executemany(
                """
                INSERT INTO reviews_fact (
                    review_id_ext, source_job_id, app_id, app_name, review_date,
                    rating, review_text_raw, review_text_clean
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                review_rows,
            )
            conn.executemany(
                """
                INSERT INTO predictions_fact (
                    review_id_ext, source_job_id, model_id, aspect, pred_label,
                    confidence, prob_negative, prob_neutral, prob_positive
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                pred_rows,
            )
            if fetch_audit_df is not None and not fetch_audit_df.empty:
                audit_rows = [
                    (
                        payload["job_id"],
                        row.app_id,
                        row.app_name,
                        int(row.stage_order),
                        row.stage_name,
                        int(row.count),
                    )
                    for row in fetch_audit_df.itertuples(index=False)
                ]
                conn.executemany(
                    """
                    INSERT INTO live_fetch_audit_fact (
                        source_job_id, app_id, app_name, stage_order, stage_name, count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    audit_rows,
                )
            conn.commit()

    def list_jobs(self) -> pd.DataFrame:
        with self.connect() as conn:
            return self._read_frame(
                conn,
                """
                SELECT job_id, app_name, app_id, date_from, date_to, review_limit,
                       fetched_at, model_id, status
                FROM review_fetch_jobs
                ORDER BY fetched_at DESC
                """,
            )

    def load_job_frames(self, job_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        review_cache_path, prediction_cache_path = self.job_cache_paths(job_id)
        if review_cache_path.exists() and prediction_cache_path.exists():
            reviews_df = pd.read_csv(review_cache_path)
            predictions_df = pd.read_csv(prediction_cache_path)
            if "source_job_id" not in reviews_df.columns:
                reviews_df["source_job_id"] = job_id
            if "source_job_id" not in predictions_df.columns:
                predictions_df["source_job_id"] = job_id
            return (reviews_df, predictions_df)

        with self.connect() as conn:
            reviews_df = self._read_frame(
                conn,
                "SELECT * FROM reviews_fact WHERE source_job_id = ?",
                params=[job_id],
            )
            predictions_df = self._read_frame(
                conn,
                "SELECT * FROM predictions_fact WHERE source_job_id = ?",
                params=[job_id],
            )
        if "source_job_id" not in reviews_df.columns:
            reviews_df["source_job_id"] = job_id
        if "source_job_id" not in predictions_df.columns:
            predictions_df["source_job_id"] = job_id
        return reviews_df, predictions_df

    def load_live_fetch_audit(self, job_id: str) -> pd.DataFrame:
        with self.connect() as conn:
            audit_df = self._read_frame(
                conn,
                """
                SELECT source_job_id, app_id, app_name, stage_order, stage_name, count
                FROM live_fetch_audit_fact
                WHERE source_job_id = ?
                ORDER BY app_name, stage_order
                """,
                params=[job_id],
            )
        return audit_df

    def upsert_registry(self, registry_df: pd.DataFrame) -> None:
        if registry_df.empty:
            return

        rows = [
            (
                row.model_id,
                row.display_name,
                row.family,
                int(row.epoch) if pd.notna(row.epoch) else None,
                row.training_regime,
                row.model_type,
                row.source_path,
                int(row.rank_weak_label) if pd.notna(row.rank_weak_label) else None,
                int(row.rank_gold_subset) if pd.notna(row.rank_gold_subset) else None,
            )
            for row in registry_df.itertuples(index=False)
        ]

        with self.connect() as conn:
            conn.execute("DELETE FROM experiment_registry")
            conn.executemany(
                """
                INSERT INTO experiment_registry (
                    model_id, display_name, family, epoch, training_regime,
                    model_type, source_path, rank_weak_label, rank_gold_subset
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def load_registry(self) -> pd.DataFrame:
        with self.connect() as conn:
            return self._read_frame(
                conn,
                "SELECT * FROM experiment_registry ORDER BY COALESCE(rank_gold_subset, 999), COALESCE(rank_weak_label, 999)",
            )

    def sync_gold_subset(self, gold_df: pd.DataFrame) -> None:
        rows = [
            (
                str(row.item_id),
                str(row.review_id),
                row.aspect,
                row.review_text,
                row.label,
                int(row.aspect_present),
                int(row.confidence) if pd.notna(row.confidence) else None,
                row.notes,
            )
            for row in gold_df.itertuples(index=False)
        ]
        with self.connect() as conn:
            conn.execute("DELETE FROM gold_subset_fact")
            conn.executemany(
                """
                INSERT INTO gold_subset_fact (
                    item_id, review_id, aspect, review_text, label,
                    aspect_present, confidence, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def sync_gold_eval(self, gold_eval_df: pd.DataFrame) -> None:
        if gold_eval_df.empty:
            return

        rows = [
            (
                row.model_id,
                str(row.item_id),
                row.pred_label,
                float(row.pred_confidence),
                int(bool(row.sentiment_match)),
            )
            for row in gold_eval_df.itertuples(index=False)
        ]
        with self.connect() as conn:
            conn.execute("DELETE FROM gold_eval_fact")
            conn.executemany(
                """
                INSERT INTO gold_eval_fact (
                    model_id, item_id, pred_label, pred_confidence, sentiment_match
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, sqlite3.Row):
            return dict(row)
        if hasattr(row, "keys"):
            return {key: row[key] for key in row.keys()}
        return dict(row)
