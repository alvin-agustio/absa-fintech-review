from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import GOLD_TEMPLATE_PATH, ROOT_DIR


GOLD_EVAL_DIRS = [
    ROOT_DIR / "data" / "processed" / "diamond" / "evaluation_all_models",
    ROOT_DIR / "data" / "processed" / "diamond" / "evaluation",
]

WEAK_EVAL_CANDIDATES = [
    ROOT_DIR / "data" / "processed" / "evaluation" / "epoch_comparison_summary.csv",
    ROOT_DIR / "droplet" / "skripsi_eval_core" / "data" / "processed" / "evaluation" / "epoch_comparison_summary.csv",
]

UNCERTAINTY_SUMMARY_CANDIDATES = [
    ROOT_DIR / "data" / "processed" / "uncertainty" / "mc_summary.json",
    ROOT_DIR / "droplet" / "skripsi_eval_core" / "data" / "processed" / "uncertainty" / "mc_summary.json",
]

NOISE_SUMMARY_CANDIDATES = [
    ROOT_DIR / "data" / "processed" / "noise" / "noise_summary.json",
    ROOT_DIR / "droplet" / "skripsi_eval_core" / "data" / "processed" / "noise" / "noise_summary.json",
]


def _resolve_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_gold_overview() -> pd.DataFrame:
    gold_dir = _resolve_first_existing(GOLD_EVAL_DIRS)
    if gold_dir is None:
        return pd.DataFrame()
    overview_path = gold_dir / "gold_evaluation_overview.csv"
    return pd.read_csv(overview_path) if overview_path.exists() else pd.DataFrame()


def load_gold_summary() -> dict:
    gold_dir = _resolve_first_existing(GOLD_EVAL_DIRS)
    if gold_dir is None:
        return {}
    summary_path = gold_dir / "gold_evaluation_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}


def load_weak_overview() -> pd.DataFrame:
    path = _resolve_first_existing(WEAK_EVAL_CANDIDATES)
    return pd.read_csv(path) if path else pd.DataFrame()


def load_uncertainty_summary() -> dict:
    path = _resolve_first_existing(UNCERTAINTY_SUMMARY_CANDIDATES)
    return json.loads(path.read_text(encoding="utf-8")) if path else {}


def load_noise_summary() -> dict:
    path = _resolve_first_existing(NOISE_SUMMARY_CANDIDATES)
    return json.loads(path.read_text(encoding="utf-8")) if path else {}


def gold_model_dir(model_id: str) -> Path | None:
    gold_dir = _resolve_first_existing(GOLD_EVAL_DIRS)
    if gold_dir is None:
        return None
    candidate = gold_dir / model_id
    return candidate if candidate.exists() else None


def load_model_gold_summary(model_id: str) -> dict:
    model_dir = gold_model_dir(model_id)
    if model_dir is None:
        return {}
    path = model_dir / "gold_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_model_gold_predictions(model_id: str) -> pd.DataFrame:
    model_dir = gold_model_dir(model_id)
    if model_dir is None:
        return pd.DataFrame()
    path = model_dir / "gold_predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_model_ladder(registry_df: pd.DataFrame) -> pd.DataFrame:
    if registry_df.empty:
        return registry_df
    ladder = registry_df.copy()
    ladder["rank_gold_subset"] = ladder["rank_gold_subset"].fillna(999)
    ladder["rank_weak_label"] = ladder["rank_weak_label"].fillna(999)
    return ladder.sort_values(["rank_gold_subset", "rank_weak_label", "display_name"])


def hardest_cases_across_models(limit: int = 20) -> pd.DataFrame:
    gold_dir = _resolve_first_existing(GOLD_EVAL_DIRS)
    if gold_dir is None:
        return pd.DataFrame()

    pred_files = sorted(gold_dir.glob("*/gold_predictions.csv"))
    merged = None
    for pred_file in pred_files:
        model_id = pred_file.parent.name
        df = pd.read_csv(pred_file)
        if df.empty:
            continue
        keep = df[["item_id", "aspect", "aspect_present", "label", "review_text", "notes", "pred_label"]].copy()
        keep.rename(columns={"pred_label": f"pred_{model_id}"}, inplace=True)
        merged = keep if merged is None else merged.merge(
            keep[["item_id", "aspect", f"pred_{model_id}"]],
            on=["item_id", "aspect"],
            how="inner",
        )

    if merged is None or merged.empty:
        return pd.DataFrame()

    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    present = merged[merged["aspect_present"] == 1].copy()
    present["error_count"] = 0
    for col in pred_cols:
        present["error_count"] += (present[col] != present["label"]).astype(int)
    return present.sort_values(["error_count", "item_id"], ascending=[False, True]).head(limit)


def absent_vote_tendency(limit: int = 20) -> pd.DataFrame:
    gold_dir = _resolve_first_existing(GOLD_EVAL_DIRS)
    if gold_dir is None:
        return pd.DataFrame()

    pred_files = sorted(gold_dir.glob("*/gold_predictions.csv"))
    merged = None
    for pred_file in pred_files:
        model_id = pred_file.parent.name
        df = pd.read_csv(pred_file)
        if df.empty:
            continue
        keep = df[["item_id", "aspect", "aspect_present", "review_text", "notes", "pred_label"]].copy()
        keep.rename(columns={"pred_label": f"pred_{model_id}"}, inplace=True)
        merged = keep if merged is None else merged.merge(
            keep[["item_id", "aspect", f"pred_{model_id}"]],
            on=["item_id", "aspect"],
            how="inner",
        )

    if merged is None or merged.empty:
        return pd.DataFrame()

    pred_cols = [col for col in merged.columns if col.startswith("pred_")]
    absent = merged[merged["aspect_present"] == 0].copy()
    absent["negative_votes"] = 0
    absent["neutral_votes"] = 0
    absent["positive_votes"] = 0
    for col in pred_cols:
        absent["negative_votes"] += (absent[col] == "Negative").astype(int)
        absent["neutral_votes"] += (absent[col] == "Neutral").astype(int)
        absent["positive_votes"] += (absent[col] == "Positive").astype(int)
    return absent.sort_values(["negative_votes", "item_id"], ascending=[False, True]).head(limit)


def build_gold_eval_fact() -> pd.DataFrame:
    gold_dir = _resolve_first_existing(GOLD_EVAL_DIRS)
    if gold_dir is None:
        return pd.DataFrame()

    rows = []
    for pred_file in gold_dir.glob("*/gold_predictions.csv"):
        model_id = pred_file.parent.name
        df = pd.read_csv(pred_file)
        if df.empty:
            continue
        part = df[["item_id", "pred_label", "pred_confidence", "sentiment_match"]].copy()
        part["model_id"] = model_id
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_gold_subset() -> pd.DataFrame:
    return pd.read_csv(GOLD_TEMPLATE_PATH)
