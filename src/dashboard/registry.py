from __future__ import annotations

import os
from pathlib import Path

import json
import pandas as pd

from config import ROOT_DIR


def _optional_env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser() if value else None


GOLD_OVERVIEW_CANDIDATES = [
    ROOT_DIR / "data" / "processed" / "diamond" / "evaluation_all_models" / "gold_evaluation_overview.csv",
    ROOT_DIR / "data" / "processed" / "diamond" / "evaluation" / "gold_evaluation_overview.csv",
]

WEAK_OVERVIEW_CANDIDATES = [
    ROOT_DIR / "data" / "processed" / "evaluation" / "epoch_comparison_summary.csv",
]

MODEL_ROOT_CANDIDATES = [
    ROOT_DIR / "models",
]

DISPLAY_NAME_MAP = {
    "baseline": "Baseline FT",
    "lora": "LoRA",
    "dora": "DoRA",
    "adalora": "AdaLoRA",
    "qlora": "QLoRA",
    "retrained": "Retrained FT",
    "retrained_lora": "Retrained LoRA",
    "retrained_dora": "Retrained DoRA",
    "retrained_adalora": "Retrained AdaLoRA",
    "retrained_qlora": "Retrained QLoRA",
}

TRAINING_REGIME_MAP = {
    "baseline": "weak_label_full_ft",
    "lora": "weak_label_lora",
    "dora": "weak_label_dora",
    "adalora": "weak_label_adalora",
    "qlora": "weak_label_qlora",
    "retrained": "clean_subset_full_ft",
    "retrained_lora": "clean_subset_lora",
    "retrained_dora": "clean_subset_dora",
    "retrained_adalora": "clean_subset_adalora",
    "retrained_qlora": "clean_subset_qlora",
}

FINAL_FAMILIES = set(DISPLAY_NAME_MAP.keys())


def resolve_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_model_paths() -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    env_root = _optional_env_path("SKRIPSI_MODEL_ROOT")
    candidates = ([env_root] if env_root else []) + MODEL_ROOT_CANDIDATES
    root = resolve_first_existing([path for path in candidates if path is not None])
    if root is None:
        return discovered

    for family_dir in root.iterdir():
        if not family_dir.is_dir():
            continue
        family = family_dir.name
        if family not in FINAL_FAMILIES:
            continue
        for epoch_dir in family_dir.iterdir():
            if not epoch_dir.is_dir() or not epoch_dir.name.startswith("epoch_"):
                continue
            epoch = epoch_dir.name.replace("epoch_", "")
            if not epoch.isdigit():
                continue
            model_dir = epoch_dir / "model"
            if not model_dir.exists():
                continue
            model_id = f"{family}_epoch_{epoch}"
            discovered.setdefault(model_id, model_dir)
    return discovered


def build_model_registry() -> pd.DataFrame:
    env_gold = _optional_env_path("SKRIPSI_GOLD_OVERVIEW_CSV")
    env_weak = _optional_env_path("SKRIPSI_WEAK_OVERVIEW_CSV")
    gold_path = resolve_first_existing(([env_gold] if env_gold else []) + GOLD_OVERVIEW_CANDIDATES)
    weak_path = resolve_first_existing(([env_weak] if env_weak else []) + WEAK_OVERVIEW_CANDIDATES)
    model_paths = discover_model_paths()

    gold_df = pd.read_csv(gold_path) if gold_path else pd.DataFrame()
    weak_df = pd.read_csv(weak_path) if weak_path else pd.DataFrame()

    rows = []
    for model_id, model_path in model_paths.items():
        family, _, epoch_text = model_id.rpartition("_epoch_")
        if family not in FINAL_FAMILIES:
            continue
        epoch = int(epoch_text)
        metrics_path = model_path.parent / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_epoch = metrics.get("best_epoch")
        best_checkpoint = metrics.get("best_checkpoint")

        gold_row = (
            gold_df[gold_df["model_name"] == model_id].iloc[0]
            if not gold_df.empty and (gold_df["model_name"] == model_id).any()
            else None
        )
        weak_row = (
            weak_df[(weak_df["model"] == family) & (weak_df["epochs"] == epoch)].iloc[0]
            if not weak_df.empty and ((weak_df["model"] == family) & (weak_df["epochs"] == epoch)).any()
            else None
        )

        rows.append(
            {
                "model_id": model_id,
                "display_name": (
                    f"{DISPLAY_NAME_MAP.get(family, family.title())} "
                    f"(run E{epoch}, best E{best_epoch})"
                    if best_epoch is not None and int(best_epoch) != epoch
                    else f"{DISPLAY_NAME_MAP.get(family, family.title())} E{epoch}"
                ),
                "family": family,
                "epoch": epoch,
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "best_checkpoint": best_checkpoint,
                "training_regime": TRAINING_REGIME_MAP.get(family, "unknown"),
                "model_type": "full_finetune" if family in {"baseline", "retrained"} else "peft",
                "source_path": str(model_path),
                "gold_f1_macro": gold_row["sentiment_f1_macro_present"] if gold_row is not None else None,
                "gold_accuracy": gold_row["sentiment_accuracy_present"] if gold_row is not None else None,
                "weak_f1_macro": weak_row["f1_macro"] if weak_row is not None else None,
                "weak_accuracy": weak_row["accuracy"] if weak_row is not None else None,
                "training_time_seconds": weak_row["training_time_seconds"] if weak_row is not None else None,
            }
        )

    registry_df = pd.DataFrame(rows)
    if registry_df.empty:
        return registry_df

    registry_df = registry_df.sort_values(
        by=["gold_f1_macro", "weak_f1_macro", "epoch"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    registry_df["rank_gold_subset"] = registry_df["gold_f1_macro"].rank(
        method="dense", ascending=False, na_option="bottom"
    )
    registry_df["rank_weak_label"] = registry_df["weak_f1_macro"].rank(
        method="dense", ascending=False, na_option="bottom"
    )
    registry_df["is_default"] = registry_df["rank_gold_subset"] == registry_df["rank_gold_subset"].min()
    return registry_df


def default_model_row(registry_df: pd.DataFrame) -> pd.Series | None:
    if registry_df.empty:
        return None
    if "is_default" in registry_df.columns and registry_df["is_default"].any():
        return registry_df[registry_df["is_default"]].iloc[0]
    return registry_df.iloc[0]
