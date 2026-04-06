from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.epoch_protocol import normalize_epoch_sweep, recommend_epochs, select_best_row


def test_selects_best_epoch_per_family_by_validation_metric():
    df = pd.DataFrame([
        {"model": "baseline", "epochs": 3, "accuracy": 0.80, "f1_macro": 0.76, "f1_weighted": 0.75, "training_time_seconds": 100, "source": "active"},
        {"model": "baseline", "epochs": 5, "accuracy": 0.82, "f1_macro": 0.78, "f1_weighted": 0.77, "training_time_seconds": 120, "source": "active"},
        {"model": "lora", "epochs": 3, "accuracy": 0.83, "f1_macro": 0.79, "f1_weighted": 0.78, "training_time_seconds": 90, "source": "active"},
        {"model": "lora", "epochs": 5, "accuracy": 0.84, "f1_macro": 0.81, "f1_weighted": 0.80, "training_time_seconds": 95, "source": "active"},
    ])

    summary = recommend_epochs(df)

    assert summary["overall_best"]["model"] == "lora"
    assert summary["overall_best"]["epochs"] == 5
    assert {row["model"]: row["epochs"] for row in summary["families"]} == {
        "baseline": 5,
        "lora": 5,
    }


def test_prefers_shorter_run_when_metrics_tie():
    df = pd.DataFrame([
        {"model": "baseline", "epochs": 3, "accuracy": 0.80, "f1_macro": 0.78, "f1_weighted": 0.77, "training_time_seconds": 100, "source": "active"},
        {"model": "baseline", "epochs": 5, "accuracy": 0.80, "f1_macro": 0.78, "f1_weighted": 0.77, "training_time_seconds": 95, "source": "active"},
    ])

    best = select_best_row(normalize_epoch_sweep(df))
    assert int(best["epochs"]) == 5


def test_prefers_active_rows_over_archived_duplicates():
    df = pd.DataFrame([
        {"model": "baseline", "epochs": 5, "accuracy": 0.81, "f1_macro": 0.79, "f1_weighted": 0.78, "training_time_seconds": 100, "source": "archived"},
        {"model": "baseline", "epochs": 5, "accuracy": 0.83, "f1_macro": 0.82, "f1_weighted": 0.81, "training_time_seconds": 90, "source": "active"},
    ])

    normalized = normalize_epoch_sweep(df)
    assert len(normalized) == 1
    best = normalized.iloc[0]
    assert best["source"] == "active"
    assert float(best["f1_macro"]) == 0.82
