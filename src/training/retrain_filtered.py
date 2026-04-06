import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from config import (
    BASE_MODEL_NAME,
    DATA_PROCESSED,
    FULL_FINETUNE_DEFAULT_LR,
    MAX_LENGTH,
    MODELS_DIR,
    SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_MAX_EPOCHS,
)
from src.training.run_utils import (
    EpochTimingCallback,
    build_epoch_log_df,
    compute_macro_metrics,
    infer_uncertainty_source,
    review_level_split,
    select_best_validation_epoch,
    write_split_manifest,
)

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ABSADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def load_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"review_id", "aspect", "review_text", "weak_label"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada clean data: {sorted(missing)}")

    work = df.copy()
    work = work[work["weak_label"].isin(LABEL2ID.keys())].copy()
    work["task_text"] = "[ASPECT=" + work["aspect"].astype(str) + "] " + work["review_text"].astype(str)
    work["label_id"] = work["weak_label"].map(LABEL2ID)
    work = work.dropna(subset=["task_text", "label_id"]).reset_index(drop=True)
    work["label_id"] = work["label_id"].astype(int)
    return work


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return compute_macro_metrics(labels, predictions)


def main():
    parser = argparse.ArgumentParser(description="Retrain IndoBERT baseline with uncertainty-filtered clean data")
    parser.add_argument("--clean_csv", default=str(DATA_PROCESSED / "noise" / "clean_data.csv"))
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    parser.add_argument("--output_dir", default=str(MODELS_DIR / "retrained"))
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=TRAIN_MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=FULL_FINETUNE_DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiment_family", default="retrained")
    parser.add_argument("--uncertainty_source_model_id", default=None)
    parser.add_argument("--noise_summary_json", default=None)
    parser.add_argument("--mc_summary_json", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_clean_data(args.clean_csv)
    if len(data) < 30:
        raise ValueError("Data clean terlalu sedikit untuk retraining.")
    uncertainty_meta = infer_uncertainty_source(args.clean_csv)

    train_df, val_df, test_df = review_level_split(
        data,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    split_manifest = write_split_manifest(output_dir, train_df, val_df, test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenizer(train_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    val_enc = tokenizer(val_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    test_enc = tokenizer(test_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)

    train_dataset = ABSADataset(train_enc, train_df["label_id"].tolist())
    val_dataset = ABSADataset(val_enc, val_df["label_id"].tolist())
    test_dataset = ABSADataset(test_enc, test_df["label_id"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
    )

    timing_callback = EpochTimingCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[timing_callback],
    )

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    epoch_log_df = build_epoch_log_df(trainer.state.log_history, timing_callback)
    if not epoch_log_df.empty:
        epoch_log_df.to_csv(output_dir / "epoch_log.csv", index=False)
    best_validation = select_best_validation_epoch(epoch_log_df) or {}

    test_output = trainer.predict(test_dataset)
    y_true = np.array(test_df["label_id"].tolist())
    y_pred = np.argmax(test_output.predictions, axis=1)
    test_metrics = compute_macro_metrics(y_true, y_pred)

    metrics = {
        "experiment_family": args.experiment_family,
        "training_regime": "full_finetune",
        "uncertainty_enabled": True,
        "uncertainty_source_model_family": uncertainty_meta["uncertainty_source_model_family"],
        "uncertainty_source_run_name": uncertainty_meta["uncertainty_source_run_name"],
        "uncertainty_source_model_id": args.uncertainty_source_model_id or uncertainty_meta["uncertainty_source_model_id"],
        "noise_summary_json": args.noise_summary_json,
        "mc_summary_json": args.mc_summary_json,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "best_epoch": int(best_validation["epoch"]) if best_validation.get("epoch") is not None else None,
        "best_checkpoint": str(trainer.state.best_model_checkpoint) if trainer.state.best_model_checkpoint else None,
        "best_validation_accuracy": best_validation.get("eval_accuracy"),
        "best_validation_precision_macro": best_validation.get("eval_precision_macro"),
        "best_validation_recall_macro": best_validation.get("eval_recall_macro"),
        "best_validation_f1_macro": best_validation.get("eval_f1_macro"),
        "best_validation_f1_weighted": best_validation.get("eval_f1_weighted"),
        "best_validation_loss": best_validation.get("eval_loss"),
        "train_start_timestamp": timing_callback.train_start_timestamp,
        "train_end_timestamp": timing_callback.train_end_timestamp,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_train_reviews": int(train_df["review_id"].nunique()),
        "n_val_reviews": int(val_df["review_id"].nunique()),
        "n_test_reviews": int(test_df["review_id"].nunique()),
        "n_total_clean": int(len(data)),
        "training_time_seconds": round(train_time, 2),
        "label_distribution": data["weak_label"].value_counts().to_dict(),
        "aspect_distribution": data["aspect"].value_counts().to_dict(),
        "split_manifest": split_manifest,
    }

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in range(3)],
        digits=4,
        zero_division=0,
    )

    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    predictions_df = test_df[["review_id", "aspect", "review_text", "weak_label"]].copy()
    predictions_df["pred_label"] = [ID2LABEL[int(x)] for x in y_pred]
    predictions_df["is_error"] = predictions_df["weak_label"] != predictions_df["pred_label"]
    predictions_df.to_csv(output_dir / "filtered_test_predictions.csv", index=False)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    with open(output_dir / "filtered_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "filtered_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("[RETRAIN] Training clean-data selesai.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
