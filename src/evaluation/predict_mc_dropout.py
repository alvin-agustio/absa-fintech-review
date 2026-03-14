import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from config import ASPECTS, DATA_PROCESSED, MAX_LENGTH, MODELS_DIR

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class InferenceDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            key: torch.tensor(values[idx]) for key, values in self.encodings.items()
        }


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
    combined["task_text"] = (
        "[ASPECT=" + combined["aspect"] + "] " + combined["review_text"].astype(str)
    )
    return combined.reset_index(drop=True)


def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(probs * np.log(probs + eps), axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Dropout inference for ABSA uncertainty"
    )
    parser.add_argument("--input_csv", default=str(DATA_PROCESSED / "dataset_absa.csv"))
    parser.add_argument("--model_dir", default=str(MODELS_DIR / "baseline" / "model"))
    parser.add_argument("--output_csv", default=str(DATA_PROCESSED / "uncertainty" / "mc_predictions.csv"))
    parser.add_argument("--num_mc", type=int, default=30)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    data = build_absa_rows(df)

    if data.empty:
        raise ValueError("Tidak ada data berlabel untuk MC Dropout inference.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

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
    output_df["pred_label"] = pred_labels
    output_df["prob_negative"] = mean_probs[:, 0]
    output_df["prob_neutral"] = mean_probs[:, 1]
    output_df["prob_positive"] = mean_probs[:, 2]
    output_df["uncertainty_entropy"] = entropy
    output_df["uncertainty_variance"] = var_probs
    output_df["is_error_vs_weak"] = output_df["pred_label"] != output_df["weak_label"]

    output_df.to_csv(output_path, index=False)

    summary = {
        "n_rows": int(len(output_df)),
        "num_mc": int(args.num_mc),
        "model_dir": str(args.model_dir),
        "mean_entropy": float(output_df["uncertainty_entropy"].mean()),
        "mean_variance": float(output_df["uncertainty_variance"].mean()),
        "error_rate_vs_weak": float(output_df["is_error_vs_weak"].mean()),
        "aspect_distribution": output_df["aspect"].value_counts().to_dict(),
    }

    with open(output_path.parent / "mc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[MC-DROPOUT] Inference selesai.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
