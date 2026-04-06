import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
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
    ID2LABEL,
    LABEL2ID,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_LENGTH,
    MODELS_DIR,
    PEFT_DEFAULT_LR,
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
    work["label"] = work["weak_label"]
    work["task_text"] = "[ASPECT=" + work["aspect"].astype(str) + "] " + work["review_text"].astype(str)
    work["label_id"] = work["label"].map(LABEL2ID)
    work = work.dropna(subset=["task_text", "label_id"]).reset_index(drop=True)
    work["label_id"] = work["label_id"].astype(int)
    return work


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return compute_macro_metrics(labels, predictions)


def main():
    parser = argparse.ArgumentParser(description="Train IndoBERT + LoRA on clean uncertainty-filtered data")
    parser.add_argument("--clean_csv", default=str(DATA_PROCESSED / "noise" / "clean_data.csv"))
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    parser.add_argument("--output_dir", default=str(MODELS_DIR / "retrained_lora"))
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=TRAIN_MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PEFT_DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--experiment_family", default="retrained_lora")
    parser.add_argument("--uncertainty_source_model_id", default=None)
    parser.add_argument("--noise_summary_json", default=None)
    parser.add_argument("--mc_summary_json", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = load_clean_data(args.clean_csv)
    if len(data) < 30:
        raise ValueError(f"Not enough clean labeled data: {len(data)} (need >= 30)")
    uncertainty_meta = infer_uncertainty_source(args.clean_csv)

    print(f"[RETRAIN-LORA] Total training samples: {len(data)}")
    print(f"  Label distribution:\n{data['label'].value_counts().to_string()}")
    print(f"  Aspect distribution:\n{data['aspect'].value_counts().to_string()}")

    train_df, val_df, test_df = review_level_split(
        data,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    split_manifest = write_split_manifest(output_path, train_df, val_df, test_df)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenizer(train_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    val_enc = tokenizer(val_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    test_enc = tokenizer(test_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)

    train_dataset = ABSADataset(train_enc, train_df["label_id"].tolist())
    val_dataset = ABSADataset(val_enc, val_df["label_id"].tolist())
    test_dataset = ABSADataset(test_enc, test_df["label_id"].tolist())

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct_trainable = trainable_params / total_params * 100

    print(f"\n[RETRAIN-LORA] LoRA Config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({pct_trainable:.2f}%)")

    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
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
        epoch_log_df.to_csv(output_path / "epoch_log.csv", index=False)
    best_validation = select_best_validation_epoch(epoch_log_df) or {}

    test_output = trainer.predict(test_dataset)
    y_true = np.array(test_df["label_id"].tolist())
    y_pred = np.argmax(test_output.predictions, axis=1)
    test_metrics = compute_macro_metrics(y_true, y_pred)

    metrics = {
        "experiment_family": args.experiment_family,
        "training_regime": "peft",
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
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_train_reviews": int(train_df["review_id"].nunique()),
        "n_val_reviews": int(val_df["review_id"].nunique()),
        "n_test_reviews": int(test_df["review_id"].nunique()),
        "n_total_clean": len(data),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(pct_trainable, 2),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "train_start_timestamp": timing_callback.train_start_timestamp,
        "train_end_timestamp": timing_callback.train_end_timestamp,
        "training_time_seconds": round(train_time, 2),
        "label_distribution": data["label"].value_counts().to_dict(),
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

    model.save_pretrained(str(output_path / "model"))
    tokenizer.save_pretrained(str(output_path / "model"))

    preds_df = test_df[["review_id", "aspect", "review_text", "weak_label"]].copy()
    preds_df["pred_label"] = [ID2LABEL[int(x)] for x in y_pred]
    preds_df["is_error"] = preds_df["weak_label"] != preds_df["pred_label"]
    preds_df.to_csv(output_path / "test_predictions.csv", index=False)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[RETRAIN-LORA] Training complete in {train_time:.1f}s")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
