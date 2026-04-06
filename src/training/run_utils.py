from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_macro_metrics(labels, predictions) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def _review_level_frame(
    data: pd.DataFrame,
    review_id_col: str = "review_id",
    label_col: str = "label_id",
) -> pd.DataFrame:
    def dominant_label(series: pd.Series) -> int:
        modes = series.mode(dropna=True)
        if len(modes) == 0:
            return int(series.iloc[0])
        return int(sorted(modes.tolist())[0])

    return (
        data.groupby(review_id_col, dropna=False)
        .agg(
            dominant_label=(label_col, dominant_label),
            n_rows=(label_col, "size"),
        )
        .reset_index()
    )


def _safe_stratify(labels: pd.Series):
    counts = labels.value_counts(dropna=False)
    return labels if not counts.empty and int(counts.min()) >= 2 else None


def review_level_split(
    data: pd.DataFrame,
    *,
    seed: int,
    test_size: float,
    val_size: float,
    review_id_col: str = "review_id",
    label_col: str = "label_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reviews = _review_level_frame(data, review_id_col=review_id_col, label_col=label_col)
    if len(reviews) < 3:
        raise ValueError("Not enough unique reviews to build train/validation/test splits.")

    train_reviews, test_reviews = train_test_split(
        reviews,
        test_size=test_size,
        random_state=seed,
        stratify=_safe_stratify(reviews["dominant_label"]),
    )
    train_reviews, val_reviews = train_test_split(
        train_reviews,
        test_size=val_size,
        random_state=seed,
        stratify=_safe_stratify(train_reviews["dominant_label"]),
    )

    train_ids = set(train_reviews[review_id_col].tolist())
    val_ids = set(val_reviews[review_id_col].tolist())
    test_ids = set(test_reviews[review_id_col].tolist())

    train_df = data[data[review_id_col].isin(train_ids)].copy()
    val_df = data[data[review_id_col].isin(val_ids)].copy()
    test_df = data[data[review_id_col].isin(test_ids)].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def write_split_manifest(
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    review_id_col: str = "review_id",
) -> dict[str, Any]:
    split_manifest = {
        "train": {
            "n_rows": int(len(train_df)),
            "n_reviews": int(train_df[review_id_col].nunique()),
        },
        "validation": {
            "n_rows": int(len(val_df)),
            "n_reviews": int(val_df[review_id_col].nunique()),
        },
        "test": {
            "n_rows": int(len(test_df)),
            "n_reviews": int(test_df[review_id_col].nunique()),
        },
    }

    manifest_df = pd.concat(
        [
            pd.DataFrame({"split": "train", review_id_col: sorted(train_df[review_id_col].astype(str).unique())}),
            pd.DataFrame({"split": "validation", review_id_col: sorted(val_df[review_id_col].astype(str).unique())}),
            pd.DataFrame({"split": "test", review_id_col: sorted(test_df[review_id_col].astype(str).unique())}),
        ],
        ignore_index=True,
    )
    manifest_df.to_csv(output_dir / "split_review_ids.csv", index=False)
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)
    return split_manifest


@dataclass
class EpochTiming:
    epoch: int
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    duration_seconds: float | None = None
    cumulative_training_seconds: float | None = None


class EpochTimingCallback(TrainerCallback):
    def __init__(self) -> None:
        self.train_start_timestamp: str | None = None
        self.train_end_timestamp: str | None = None
        self._train_start_perf: float | None = None
        self._epoch_start_perf: dict[int, float] = {}
        self.epoch_timings: dict[int, EpochTiming] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        import time

        self.train_start_timestamp = iso_utc_now()
        self._train_start_perf = time.perf_counter()

    def on_epoch_begin(self, args, state, control, **kwargs):
        import time

        epoch = int(round(float(state.epoch or 0.0))) + 1
        self._epoch_start_perf[epoch] = time.perf_counter()
        timing = self.epoch_timings.get(epoch, EpochTiming(epoch=epoch))
        timing.start_timestamp = iso_utc_now()
        self.epoch_timings[epoch] = timing

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import time

        epoch = int(round(float(state.epoch or 0.0)))
        start_perf = self._epoch_start_perf.get(epoch)
        timing = self.epoch_timings.get(epoch, EpochTiming(epoch=epoch))
        timing.end_timestamp = iso_utc_now()
        if start_perf is not None:
            timing.duration_seconds = round(time.perf_counter() - start_perf, 4)
        if self._train_start_perf is not None:
            timing.cumulative_training_seconds = round(time.perf_counter() - self._train_start_perf, 4)
        self.epoch_timings[epoch] = timing

    def on_train_end(self, args, state, control, **kwargs):
        self.train_end_timestamp = iso_utc_now()


def build_epoch_log_df(log_history: list[dict[str, Any]], timing_callback: EpochTimingCallback) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in log_history:
        if "eval_f1_macro" not in item:
            continue
        epoch = int(round(float(item.get("epoch", 0.0))))
        timing = timing_callback.epoch_timings.get(epoch, EpochTiming(epoch=epoch))
        rows.append(
            {
                "epoch": epoch,
                "eval_accuracy": float(item.get("eval_accuracy")) if item.get("eval_accuracy") is not None else None,
                "eval_precision_macro": float(item.get("eval_precision_macro")) if item.get("eval_precision_macro") is not None else None,
                "eval_recall_macro": float(item.get("eval_recall_macro")) if item.get("eval_recall_macro") is not None else None,
                "eval_f1_macro": float(item.get("eval_f1_macro")) if item.get("eval_f1_macro") is not None else None,
                "eval_f1_weighted": float(item.get("eval_f1_weighted")) if item.get("eval_f1_weighted") is not None else None,
                "eval_loss": float(item.get("eval_loss")) if item.get("eval_loss") is not None else None,
                "eval_runtime": float(item.get("eval_runtime")) if item.get("eval_runtime") is not None else None,
                "eval_samples_per_second": float(item.get("eval_samples_per_second")) if item.get("eval_samples_per_second") is not None else None,
                "eval_steps_per_second": float(item.get("eval_steps_per_second")) if item.get("eval_steps_per_second") is not None else None,
                "step": int(item.get("step")) if item.get("step") is not None else None,
                "epoch_start_timestamp": timing.start_timestamp,
                "epoch_end_timestamp": timing.end_timestamp,
                "epoch_duration_seconds": timing.duration_seconds,
                "cumulative_training_seconds": timing.cumulative_training_seconds,
            }
        )
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True) if rows else pd.DataFrame()


def select_best_validation_epoch(epoch_log_df: pd.DataFrame) -> dict[str, Any] | None:
    if epoch_log_df.empty:
        return None

    ranked = epoch_log_df.sort_values(
        by=[
            "eval_f1_macro",
            "eval_accuracy",
            "eval_precision_macro",
            "eval_recall_macro",
            "cumulative_training_seconds",
            "epoch",
        ],
        ascending=[False, False, False, False, True, True],
        na_position="last",
    )
    best = ranked.iloc[0].to_dict()
    return best


def infer_uncertainty_source(clean_csv_path: str | Path | None) -> dict[str, Any]:
    if not clean_csv_path:
        return {
            "uncertainty_enabled": False,
            "uncertainty_source_model_family": None,
            "uncertainty_source_run_name": None,
            "uncertainty_source_model_id": None,
        }

    path = Path(clean_csv_path)
    parts = list(path.parts)
    if "noise" not in parts:
        return {
            "uncertainty_enabled": False,
            "uncertainty_source_model_family": None,
            "uncertainty_source_run_name": None,
            "uncertainty_source_model_id": None,
        }

    noise_idx = parts.index("noise")
    if len(parts) <= noise_idx + 2:
        return {
            "uncertainty_enabled": True,
            "uncertainty_source_model_family": None,
            "uncertainty_source_run_name": None,
            "uncertainty_source_model_id": None,
        }

    family = parts[noise_idx + 1]
    run_name = parts[noise_idx + 2] if len(parts) > noise_idx + 2 else None
    return {
        "uncertainty_enabled": True,
        "uncertainty_source_model_family": family,
        "uncertainty_source_run_name": run_name,
        "uncertainty_source_model_id": f"{family}:{run_name}" if family and run_name else None,
    }
