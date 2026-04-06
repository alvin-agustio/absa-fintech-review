from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PRIMARY_METRIC = "f1_macro"
REQUIRED_COLUMNS = {"model", "epochs", "accuracy", "f1_macro", "f1_weighted"}
METRIC_FALLBACK_ORDER = ["f1_macro", "accuracy", "f1_weighted", "training_time_seconds", "epochs"]
SOURCE_PRIORITY = {"active": 0}


def load_epoch_sweep(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Epoch sweep CSV not found: {csv_path}")
    return normalize_epoch_sweep(pd.read_csv(csv_path))


def normalize_epoch_sweep(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    missing = REQUIRED_COLUMNS.difference(work.columns)
    if missing:
        raise ValueError(f"Epoch sweep is missing required columns: {sorted(missing)}")

    if "source" not in work.columns:
        work["source"] = "active"
    if "training_time_seconds" not in work.columns:
        work["training_time_seconds"] = pd.NA

    work["model"] = work["model"].astype(str)
    work["source"] = work["source"].astype(str)
    work["epochs"] = pd.to_numeric(work["epochs"], errors="raise").astype(int)

    for column in ["accuracy", "f1_macro", "f1_weighted", "training_time_seconds"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work["source_priority"] = work["source"].map(lambda value: SOURCE_PRIORITY.get(value, 1)).astype(int)
    work = (
        work.sort_values(["model", "epochs", "source_priority"])
        .drop_duplicates(subset=["model", "epochs"], keep="first")
        .reset_index(drop=True)
    )
    return work


def _ordered_metric_columns(df: pd.DataFrame, primary_metric: str) -> list[str]:
    ordered = [primary_metric]
    for column in METRIC_FALLBACK_ORDER:
        if column not in ordered and column in df.columns:
            ordered.append(column)
    return ordered


def select_best_row(df: pd.DataFrame, primary_metric: str = PRIMARY_METRIC) -> pd.Series:
    if df.empty:
        raise ValueError("Cannot select a best row from an empty dataframe.")
    if primary_metric not in df.columns:
        raise ValueError(f"Primary metric not found in epoch sweep: {primary_metric}")

    sort_columns = _ordered_metric_columns(df, primary_metric)
    ascending = [False if column not in {"training_time_seconds", "epochs"} else True for column in sort_columns]

    if "source_priority" in df.columns and "source_priority" not in sort_columns:
        sort_columns.append("source_priority")
        ascending.append(True)

    ranked = df.sort_values(sort_columns, ascending=ascending, na_position="last")
    return ranked.iloc[0]


def recommend_epochs(df: pd.DataFrame, primary_metric: str = PRIMARY_METRIC) -> dict[str, Any]:
    normalized = normalize_epoch_sweep(df)
    if normalized.empty:
        raise ValueError("Epoch sweep dataframe is empty.")

    family_rows: list[dict[str, Any]] = []
    for family, family_df in normalized.groupby("model", sort=True):
        best = select_best_row(family_df, primary_metric=primary_metric)
        family_rows.append({
            "model": family,
            "epochs": int(best["epochs"]),
            "accuracy": float(best["accuracy"]),
            "f1_macro": float(best["f1_macro"]),
            "f1_weighted": float(best["f1_weighted"]),
            "training_time_seconds": float(best["training_time_seconds"]) if pd.notna(best["training_time_seconds"]) else None,
            "source": str(best.get("source", "active")),
        })

    overall_best = select_best_row(normalized, primary_metric=primary_metric)
    return {
        "primary_metric": primary_metric,
        "n_rows": int(len(normalized)),
        "families": family_rows,
        "overall_best": {
            "model": str(overall_best["model"]),
            "epochs": int(overall_best["epochs"]),
            "accuracy": float(overall_best["accuracy"]),
            "f1_macro": float(overall_best["f1_macro"]),
            "f1_weighted": float(overall_best["f1_weighted"]),
            "training_time_seconds": float(overall_best["training_time_seconds"]) if pd.notna(overall_best["training_time_seconds"]) else None,
            "source": str(overall_best.get("source", "active")),
        },
    }


def format_recommendation_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Epoch Recommendation",
        "",
        f"Primary metric: `{summary['primary_metric']}`",
        f"Rows analyzed: `{summary['n_rows']}`",
        "",
        "## Per Family",
    ]

    for row in summary["families"]:
        time_display = "N/A" if row["training_time_seconds"] is None else f"{row['training_time_seconds']:.2f}s"
        lines.append(
            f"- `{row['model']}` -> epoch `{row['epochs']}` "
            f"(F1 Macro `{row['f1_macro']:.4f}`, Accuracy `{row['accuracy']:.4f}`, Time `{time_display}`)"
        )

    overall = summary["overall_best"]
    time_display = "N/A" if overall["training_time_seconds"] is None else f"{overall['training_time_seconds']:.2f}s"
    lines.extend([
        "",
        "## Overall Best",
        f"- `{overall['model']}` epoch `{overall['epochs']}`",
        f"- F1 Macro `{overall['f1_macro']:.4f}`",
        f"- Accuracy `{overall['accuracy']:.4f}`",
        f"- Time `{time_display}`",
    ])
    return "\n".join(lines)
