import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover
    PeftConfig = None
    PeftModel = None

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from config import ASPECTS, BASE_MODEL_NAME, DATA_PROCESSED, MAX_LENGTH, MODELS_DIR

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class InferenceDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}


def build_absa_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    aspect_col_map = {
        "risk": "risk_sentiment",
        "trust": "trust_sentiment",
        "service": "service_sentiment",
    }

    for aspect in ASPECTS:
        col = aspect_col_map[aspect]
        if col not in df.columns:
            continue
        subset = df[df[col].isin(LABEL2ID.keys())][["review_id", "review_text", col]].copy()
        subset["aspect"] = aspect
        subset.rename(columns={col: "weak_label"}, inplace=True)
        rows.append(subset)

    combined = pd.concat(rows, ignore_index=True)
    combined["task_text"] = "[ASPECT=" + combined["aspect"] + "] " + combined["review_text"].astype(str)
    return combined.reset_index(drop=True)


def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(probs * np.log(probs + eps), axis=1)


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_model_family(model_dir: Path, explicit_family: str | None) -> str:
    if explicit_family:
        return explicit_family.strip().lower().replace(" ", "_")

    lowered = str(model_dir).lower()
    if "adalora" in lowered:
        return "adalora"
    if "qlora" in lowered:
        return "qlora"
    if "dora" in lowered:
        return "dora"
    if "retrained_lora" in lowered:
        return "retrained_lora"
    if "retrained" in lowered:
        return "retrained"
    if "lora" in lowered:
        return "lora"
    return "baseline"


def infer_run_name(model_dir: Path, explicit_run_name: str | None) -> str:
    if explicit_run_name:
        return explicit_run_name.strip().replace(" ", "_")
    for part in reversed(model_dir.parts):
        if part.startswith("epoch_"):
            return part
    return model_dir.name.replace(" ", "_")


def load_model_and_tokenizer(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if (model_dir / "adapter_config.json").exists():
        if PeftConfig is None or PeftModel is None:
            raise ImportError("PEFT is required to load adapter-based model directories.")
        peft_config = PeftConfig.from_pretrained(str(model_dir))
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        model_type = "peft"
        base_model_name = peft_config.base_model_name_or_path
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model_type = "full_finetune"
        base_model_name = getattr(model.config, "_name_or_path", BASE_MODEL_NAME)

    return model, tokenizer, model_type, base_model_name


def count_dropout_modules(model: torch.nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, torch.nn.Dropout))


def resolve_output_paths(
    *,
    output_csv: str | None,
    output_dir: str | None,
    model_family: str,
    run_name: str,
) -> tuple[Path, Path]:
    if output_csv:
        csv_path = Path(output_csv)
        return csv_path, csv_path.parent

    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = DATA_PROCESSED / "uncertainty"

    final_dir = base_dir / model_family / run_name
    return final_dir / "mc_predictions.csv", final_dir


def summarize_by_aspect(output_df: pd.DataFrame) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for aspect, part in output_df.groupby("aspect"):
        summary[str(aspect)] = {
            "n_rows": int(len(part)),
            "mean_entropy": float(part["uncertainty_entropy"].mean()),
            "mean_variance": float(part["uncertainty_variance"].mean()),
            "error_rate_vs_weak": float(part["is_error_vs_weak"].mean()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Dropout inference for ABSA uncertainty")
    parser.add_argument("--input_csv", default=str(DATA_PROCESSED / "dataset_absa.csv"))
    parser.add_argument("--model_dir", default=str(MODELS_DIR / "baseline" / "model"))
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_family", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--num_mc", type=int, default=30)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    model_family = infer_model_family(model_dir, args.model_family)
    run_name = infer_run_name(model_dir, args.run_name)
    output_csv_path, output_dir = resolve_output_paths(
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        model_family=model_family,
        run_name=run_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    data = build_absa_rows(df)
    if data.empty:
        raise ValueError("Tidak ada data berlabel untuk MC Dropout inference.")

    model, tokenizer, model_type, base_model_name = load_model_and_tokenizer(model_dir)
    dropout_modules = count_dropout_modules(model)
    if dropout_modules == 0:
        raise ValueError(f"No dropout modules detected in model_dir={model_dir}. MC Dropout would be invalid.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encodings = tokenizer(
        data["task_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    dataset = InferenceDataset(encodings)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    mc_probs = []
    model.train()
    for _ in range(args.num_mc):
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        mc_probs.append(np.vstack(all_probs))

    mc_probs = np.stack(mc_probs, axis=0)
    mean_probs = mc_probs.mean(axis=0)
    var_probs = mc_probs.var(axis=0).mean(axis=1)
    entropy = entropy_from_probs(mean_probs)

    pred_ids = mean_probs.argmax(axis=1)
    pred_labels = [ID2LABEL[int(x)] for x in pred_ids]

    output_df = data[["review_id", "aspect", "review_text", "weak_label"]].copy()
    output_df["model_family"] = model_family
    output_df["run_name"] = run_name
    output_df["model_dir"] = str(model_dir)
    output_df["model_type"] = model_type
    output_df["pred_label"] = pred_labels
    output_df["pred_confidence"] = mean_probs.max(axis=1)
    output_df["prob_negative"] = mean_probs[:, 0]
    output_df["prob_neutral"] = mean_probs[:, 1]
    output_df["prob_positive"] = mean_probs[:, 2]
    output_df["uncertainty_entropy"] = entropy
    output_df["uncertainty_variance"] = var_probs
    output_df["is_error_vs_weak"] = output_df["pred_label"] != output_df["weak_label"]

    output_df.to_csv(output_csv_path, index=False)

    summary = {
        "n_rows": int(len(output_df)),
        "model_family": model_family,
        "run_name": run_name,
        "model_dir": str(model_dir),
        "model_type": model_type,
        "base_model_name": base_model_name,
        "num_mc": int(args.num_mc),
        "dropout_modules": int(dropout_modules),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "generated_at_utc": iso_utc_now(),
        "mean_entropy": float(output_df["uncertainty_entropy"].mean()),
        "mean_variance": float(output_df["uncertainty_variance"].mean()),
        "error_rate_vs_weak": float(output_df["is_error_vs_weak"].mean()),
        "pred_confidence_mean": float(output_df["pred_confidence"].mean()),
        "entropy_quantiles": {
            "q50": float(output_df["uncertainty_entropy"].quantile(0.5)),
            "q80": float(output_df["uncertainty_entropy"].quantile(0.8)),
            "q90": float(output_df["uncertainty_entropy"].quantile(0.9)),
            "q95": float(output_df["uncertainty_entropy"].quantile(0.95)),
        },
        "variance_quantiles": {
            "q50": float(output_df["uncertainty_variance"].quantile(0.5)),
            "q80": float(output_df["uncertainty_variance"].quantile(0.8)),
            "q90": float(output_df["uncertainty_variance"].quantile(0.9)),
            "q95": float(output_df["uncertainty_variance"].quantile(0.95)),
        },
        "aspect_distribution": output_df["aspect"].value_counts().to_dict(),
        "aspect_stats": summarize_by_aspect(output_df),
        "output_csv": str(output_csv_path),
    }

    with open(output_dir / "mc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[MC-DROPOUT] Inference selesai.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
