import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GOLD_TEMPLATE_PATH, ID2LABEL, LABEL2ID, MAX_LENGTH, ROOT_DIR


LABEL_ORDER = ["Negative", "Neutral", "Positive"]
DEFAULT_THRESHOLDS = [0.5]
PEFT_HINTS = ("lora", "dora", "qlora", "adalora")


def default_model_specs() -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    active_models_root = ROOT_DIR / "models"
    if active_models_root.exists():
        for family_dir in sorted(active_models_root.iterdir()):
            if not family_dir.is_dir() or family_dir.name == "archive":
                continue
            for epoch_dir in sorted(family_dir.glob("epoch_*")):
                model_dir = epoch_dir / "model"
                metrics_path = epoch_dir / "metrics.json"
                if model_dir.exists() and metrics_path.exists():
                    discovered.append((f"{family_dir.name}_{epoch_dir.name}", model_dir))
        if discovered:
            return discovered

    candidates = [
        (
            "baseline_epoch5",
            [
                ROOT_DIR
                / "droplet"
                / "skripsi_post_training_all_no_checkpoints_2026-03-13_165411"
                / "models"
                / "baseline"
                / "epoch_5"
                / "model",
                ROOT_DIR / "models" / "baseline" / "epoch_5" / "model",
                ROOT_DIR / "models" / "baseline" / "model",
            ],
        ),
        (
            "retrained_epoch8",
            [
                ROOT_DIR
                / "droplet"
                / "skripsi_post_training_all_no_checkpoints_2026-03-13_165411"
                / "models"
                / "retrained"
                / "epoch_8"
                / "model",
                ROOT_DIR / "models" / "retrained" / "epoch_8" / "model",
                ROOT_DIR / "models" / "retrained" / "model",
            ],
        ),
        (
            "retrained_lora_epoch8",
            [
                ROOT_DIR
                / "droplet"
                / "skripsi_post_training_all_no_checkpoints_2026-03-13_165411"
                / "models"
                / "retrained_lora"
                / "epoch_8"
                / "model",
                ROOT_DIR
                / "droplet"
                / "skripsi_best_model_retrained_lora_epoch8"
                / "models"
                / "retrained_lora"
                / "epoch_8"
                / "model",
                ROOT_DIR / "models" / "retrained_lora" / "epoch_8" / "model",
            ],
        ),
    ]

    resolved = []
    for name, paths in candidates:
        for path in paths:
            if path.exists():
                resolved.append((name, path))
                break
    return resolved


def parse_model_specs(specs: list[str] | None) -> list[tuple[str, Path]]:
    if not specs:
        return default_model_specs()

    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --model_spec '{spec}'. Use name=/path/to/model.")
        name, raw_path = spec.split("=", 1)
        parsed.append((name.strip(), Path(raw_path).expanduser().resolve()))
    return parsed


def infer_model_metadata(name: str) -> dict:
    lowered = name.lower()
    training_regime = "peft" if any(token in lowered for token in PEFT_HINTS) else "full_finetune"
    uncertainty_enabled = ("retrained" in lowered) or lowered.endswith("_unc") or "_unc_" in lowered
    comparison_group = "baseline"
    if training_regime == "peft":
        comparison_group = "peft"
    if uncertainty_enabled and training_regime == "full_finetune":
        comparison_group = "baseline_uncertainty"
    elif uncertainty_enabled and training_regime == "peft":
        comparison_group = "peft_uncertainty"
    return {
        "training_regime": training_regime,
        "uncertainty_enabled": uncertainty_enabled,
        "comparison_group": comparison_group,
    }


def load_gold_subset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"item_id", "review_id", "aspect", "review_text", "label", "aspect_present"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Gold subset missing required columns: {sorted(missing)}")

    work = df.copy()
    work["review_text"] = work["review_text"].fillna("").astype(str)
    work["aspect"] = work["aspect"].fillna("").astype(str)
    work["label"] = work["label"].fillna("").astype(str)
    work["aspect_present"] = work["aspect_present"].fillna(0).astype(int)
    work["task_text"] = "[ASPECT=" + work["aspect"] + "] " + work["review_text"]
    return work


def load_model_and_tokenizer(model_dir: Path, local_files_only: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=local_files_only)

    if (model_dir / "adapter_config.json").exists():
        peft_config = PeftConfig.from_pretrained(str(model_dir), local_files_only=local_files_only)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(model_dir),
            local_files_only=local_files_only,
        )
        model_type = "peft"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=local_files_only,
        )
        model_type = "full_finetune"

    return model, tokenizer, model_type


def batched_predict(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()

    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        logits_batches.append(logits.cpu().numpy())
        prob_batches.append(probs.cpu().numpy())

    logits_all = np.concatenate(logits_batches, axis=0)
    probs_all = np.concatenate(prob_batches, axis=0)
    return logits_all, probs_all


def round_float(value):
    return round(float(value), 4)


def classification_report_dict(y_true, y_pred) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    clean = {}
    for key, val in report.items():
        if isinstance(val, dict):
            clean[key] = {k: round_float(v) for k, v in val.items()}
        else:
            clean[key] = round_float(val)
    return clean


def sentiment_metrics_present_only(df: pd.DataFrame) -> dict:
    present_df = df[df["aspect_present"] == 1].copy()
    if present_df.empty:
        raise ValueError("Gold subset has no rows with aspect_present = 1.")

    y_true = present_df["label"]
    y_pred = present_df["pred_label"]
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    per_aspect = {}
    for aspect, aspect_df in present_df.groupby("aspect"):
        per_aspect[aspect] = {
            "n_rows": int(len(aspect_df)),
            "accuracy": round_float(accuracy_score(aspect_df["label"], aspect_df["pred_label"])),
            "f1_macro": round_float(
                f1_score(
                    aspect_df["label"],
                    aspect_df["pred_label"],
                    average="macro",
                    labels=LABEL_ORDER,
                    zero_division=0,
                )
            ),
            "f1_weighted": round_float(
                f1_score(
                    aspect_df["label"],
                    aspect_df["pred_label"],
                    average="weighted",
                    labels=LABEL_ORDER,
                    zero_division=0,
                )
            ),
        }

    return {
        "n_rows": int(len(present_df)),
        "accuracy": round_float(accuracy_score(y_true, y_pred)),
        "f1_macro": round_float(f1_score(y_true, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0)),
        "f1_weighted": round_float(
            f1_score(y_true, y_pred, average="weighted", labels=LABEL_ORDER, zero_division=0)
        ),
        "classification_report": classification_report_dict(y_true, y_pred),
        "confusion_matrix": {
            actual: {pred: int(cm[i][j]) for j, pred in enumerate(LABEL_ORDER)}
            for i, actual in enumerate(LABEL_ORDER)
        },
        "per_aspect": per_aspect,
    }


def absent_row_diagnostics(df: pd.DataFrame) -> dict:
    absent_df = df[df["aspect_present"] == 0].copy()
    if absent_df.empty:
        return {"n_rows": 0}

    return {
        "n_rows": int(len(absent_df)),
        "predicted_label_distribution": {
            str(k): int(v) for k, v in absent_df["pred_label"].value_counts().to_dict().items()
        },
        "mean_confidence": round_float(absent_df["pred_confidence"].mean()),
        "median_confidence": round_float(absent_df["pred_confidence"].median()),
        "max_confidence": round_float(absent_df["pred_confidence"].max()),
        "top_confident_absent_cases": absent_df.sort_values("pred_confidence", ascending=False)
        .head(10)[["item_id", "review_id", "aspect", "review_text", "pred_label", "pred_confidence", "notes"]]
        .to_dict("records"),
    }


def aspect_presence_metrics(df: pd.DataFrame, thresholds: list[float]) -> list[dict]:
    y_true = df["aspect_present"].astype(int).to_numpy()
    results = []

    for threshold in thresholds:
        y_pred = (df["pred_confidence"] >= threshold).astype(int).to_numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        results.append(
            {
                "threshold": threshold,
                "accuracy": round_float(accuracy_score(y_true, y_pred)),
                "precision": round_float(precision),
                "recall": round_float(recall),
                "f1": round_float(f1),
                "confusion_matrix": {
                    "gold_absent": {"pred_absent": int(cm[0][0]), "pred_present": int(cm[0][1])},
                    "gold_present": {"pred_absent": int(cm[1][0]), "pred_present": int(cm[1][1])},
                },
            }
        )

    return results


def evaluate_single_model(
    name: str,
    model_dir: Path,
    gold_df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    device: torch.device,
    thresholds: list[float],
    output_dir: Path,
    local_files_only: bool,
) -> dict:
    model, tokenizer, model_type = load_model_and_tokenizer(model_dir, local_files_only=local_files_only)
    _, probs = batched_predict(
        model=model,
        tokenizer=tokenizer,
        texts=gold_df["task_text"].tolist(),
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    pred_ids = probs.argmax(axis=1)
    pred_labels = [ID2LABEL[int(idx)] for idx in pred_ids]
    pred_conf = probs.max(axis=1)

    result_df = gold_df.copy()
    result_df["pred_label"] = pred_labels
    result_df["pred_confidence"] = pred_conf
    result_df["prob_negative"] = probs[:, 0]
    result_df["prob_neutral"] = probs[:, 1]
    result_df["prob_positive"] = probs[:, 2]
    result_df["gold_label_effective"] = np.where(result_df["aspect_present"] == 1, result_df["label"], "")
    result_df["sentiment_match"] = np.where(
        result_df["aspect_present"] == 1,
        result_df["label"] == result_df["pred_label"],
        False,
    )

    model_out_dir = output_dir / name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(model_out_dir / "gold_predictions.csv", index=False, encoding="utf-8")

    summary = {
        "model_name": name,
        **infer_model_metadata(name),
        "model_dir": str(model_dir),
        "model_type": model_type,
        "n_rows": int(len(result_df)),
        "n_present": int((result_df["aspect_present"] == 1).sum()),
        "n_absent": int((result_df["aspect_present"] == 0).sum()),
        "sentiment_present_only": sentiment_metrics_present_only(result_df),
        "aspect_absent_diagnostics": absent_row_diagnostics(result_df),
        "aspect_presence_threshold_sweep": aspect_presence_metrics(result_df, thresholds),
    }

    with open(model_out_dir / "gold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def make_overview_table(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        best_presence = max(summary["aspect_presence_threshold_sweep"], key=lambda item: item["f1"])
        model_name = summary["model_name"]
        family = model_name.rsplit("_epoch", 1)[0] if "_epoch" in model_name else model_name
        uncertainty_variant = (
            "with_uncertainty"
            if any(token in family for token in ["retrained", "_unc", "uncertainty"])
            else "without_uncertainty"
        )
        rows.append(
            {
                "model_name": summary["model_name"],
                "family": family,
                "uncertainty_variant": uncertainty_variant,
                "training_regime": summary.get("training_regime"),
                "uncertainty_enabled": summary.get("uncertainty_enabled"),
                "comparison_group": summary.get("comparison_group"),
                "model_type": summary["model_type"],
                "n_present": summary["n_present"],
                "n_absent": summary["n_absent"],
                "sentiment_accuracy_present": summary["sentiment_present_only"]["accuracy"],
                "sentiment_f1_macro_present": summary["sentiment_present_only"]["f1_macro"],
                "sentiment_f1_weighted_present": summary["sentiment_present_only"]["f1_weighted"],
                "best_presence_threshold": best_presence["threshold"],
                "best_presence_f1": best_presence["f1"],
                "best_presence_precision": best_presence["precision"],
                "best_presence_recall": best_presence["recall"],
                "absent_mean_confidence": summary["aspect_absent_diagnostics"].get("mean_confidence"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["sentiment_f1_macro_present", "best_presence_f1"],
        ascending=[False, False],
    )


def make_group_best_table(overview_df: pd.DataFrame) -> pd.DataFrame:
    if overview_df.empty:
        return overview_df
    ranked = overview_df.sort_values(
        by=["comparison_group", "sentiment_f1_macro_present", "best_presence_f1"],
        ascending=[True, False, False],
    )
    return ranked.groupby("comparison_group", as_index=False).head(1).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate best ABSA models on the gold manual subset.")
    parser.add_argument("--gold_csv", default=str(GOLD_TEMPLATE_PATH))
    parser.add_argument("--output_dir", default=str(Path(GOLD_TEMPLATE_PATH).parent / "evaluation"))
    parser.add_argument("--model_spec", action="append", help="Repeatable: name=/abs/or/relative/model_dir")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--presence_threshold_mode", choices=["fixed", "sweep"], default="fixed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = args.thresholds if args.presence_threshold_mode == "sweep" else [args.thresholds[0]]

    gold_df = load_gold_subset(gold_path)
    model_specs = parse_model_specs(args.model_spec)
    if not model_specs:
        raise ValueError("No model directories found. Pass --model_spec name=/path/to/model.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    gold_meta = {
        "gold_csv": str(gold_path),
        "n_rows": int(len(gold_df)),
        "n_present": int((gold_df["aspect_present"] == 1).sum()),
        "n_absent": int((gold_df["aspect_present"] == 0).sum()),
        "aspect_distribution": {str(k): int(v) for k, v in gold_df["aspect"].value_counts().to_dict().items()},
        "present_label_distribution": {
            str(k): int(v)
            for k, v in gold_df[gold_df["aspect_present"] == 1]["label"].value_counts().to_dict().items()
        },
    }

    summaries = []
    failures = []
    for name, model_dir in model_specs:
        if not model_dir.exists():
            failures.append({"model_name": name, "model_dir": str(model_dir), "error": "model directory not found"})
            continue
        try:
            print(f"[GOLD] Evaluating {name} -> {model_dir}")
            summaries.append(
                evaluate_single_model(
                    name=name,
                    model_dir=model_dir,
                    gold_df=gold_df,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=device,
                    thresholds=thresholds,
                    output_dir=output_dir,
                    local_files_only=args.local_files_only,
                )
            )
        except Exception as exc:
            failures.append({"model_name": name, "model_dir": str(model_dir), "error": str(exc)})

    overview_df = make_overview_table(summaries) if summaries else pd.DataFrame()
    if not overview_df.empty:
        overview_df.to_csv(output_dir / "gold_evaluation_overview.csv", index=False, encoding="utf-8")
        group_best_df = make_group_best_table(overview_df)
        group_best_df.to_csv(output_dir / "gold_evaluation_group_best.csv", index=False, encoding="utf-8")

    payload = {
        "gold_subset": gold_meta,
        "presence_threshold_mode": args.presence_threshold_mode,
        "thresholds_used": thresholds,
        "models_evaluated": summaries,
        "failures": failures,
    }

    with open(output_dir / "gold_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("[GOLD] Gold subset summary:")
    print(json.dumps(gold_meta, indent=2, ensure_ascii=False))

    if not overview_df.empty:
        print("\n[GOLD] Overview:")
        print(overview_df.to_string(index=False))

    if failures:
        print("\n[GOLD] Failures:")
        print(json.dumps(failures, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
