import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
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
    SEED,
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
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train IndoBERT + LoRA on clean uncertainty-filtered data")
    parser.add_argument("--clean_csv", default=str(DATA_PROCESSED / "noise" / "clean_data.csv"))
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    parser.add_argument("--output_dir", default=str(MODELS_DIR / "retrained_lora"))
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    args = parser.parse_args()

    set_seed(args.seed)
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = load_clean_data(args.clean_csv)

    if len(data) < 30:
        raise ValueError(f"Not enough clean labeled data: {len(data)} (need >= 30)")

    print(f"[RETRAIN-LORA] Total training samples: {len(data)}")
    print(f"  Label distribution:\n{data['label'].value_counts().to_string()}")
    print(f"  Aspect distribution:\n{data['aspect'].value_counts().to_string()}")

    train_df, test_df = train_test_split(
        data, test_size=args.test_size, random_state=args.seed, stratify=data["label_id"],
    )
    train_df, val_df = train_test_split(
        train_df, test_size=args.val_size, random_state=args.seed, stratify=train_df["label_id"],
    )

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenizer(train_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    val_enc = tokenizer(val_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    test_enc = tokenizer(test_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)

    train_dataset = ABSADataset(train_enc, train_df["label_id"].tolist())
    val_dataset = ABSADataset(val_enc, val_df["label_id"].tolist())
    test_dataset = ABSADataset(test_enc, test_df["label_id"].tolist())

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    test_output = trainer.predict(test_dataset)
    y_true = np.array(test_df["label_id"].tolist())
    y_pred = np.argmax(test_output.predictions, axis=1)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_total_clean": len(data),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(pct_trainable, 2),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "training_time_seconds": round(train_time, 2),
        "label_distribution": data["label"].value_counts().to_dict(),
        "aspect_distribution": data["aspect"].value_counts().to_dict(),
    }

    report_text = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in range(3)],
        digits=4, zero_division=0,
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
