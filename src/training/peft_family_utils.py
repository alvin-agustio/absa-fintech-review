from __future__ import annotations

import json
import math
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from config import (
    ADALORA_ALPHA,
    ADALORA_BETA1,
    ADALORA_BETA2,
    ADALORA_DELTA_T,
    ADALORA_DROPOUT,
    ADALORA_INIT_R,
    ADALORA_INIT_R_CANDIDATES,
    ADALORA_ORTH_REG_WEIGHT,
    ADALORA_TARGET_MODULES,
    ADALORA_TARGET_R,
    ADALORA_TFINAL,
    ADALORA_TINIT,
    BASE_MODEL_NAME,
    DATA_PROCESSED,
    DORA_ALPHA,
    DORA_DROPOUT,
    DORA_R,
    DORA_TARGET_MODULES,
    ID2LABEL,
    LABEL2ID,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_LENGTH,
    MODELS_DIR,
    PEFT_DEFAULT_LR,
    FULL_FINETUNE_DEFAULT_LR,
    QLORA_ALPHA,
    QLORA_COMPUTE_DTYPE,
    QLORA_DOUBLE_QUANT,
    QLORA_DROPOUT,
    QLORA_QUANT_TYPE,
    QLORA_R,
    QLORA_TARGET_MODULES,
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


class AdaLoraStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if hasattr(model, "update_and_allocate"):
            model.update_and_allocate(state.global_step)
        elif hasattr(model, "base_model") and hasattr(model.base_model, "update_and_allocate"):
            model.base_model.update_and_allocate(state.global_step)


@dataclass(frozen=True)
class PeftFamilySpec:
    family_name: str
    display_name: str
    target_modules: list[str]
    r: int
    alpha: int
    dropout: float
    use_dora: bool = False
    use_adalora: bool = False
    use_qlora: bool = False


FAMILY_SPECS: dict[str, PeftFamilySpec] = {
    "lora": PeftFamilySpec("lora", "LoRA", list(LORA_TARGET_MODULES), LORA_R, LORA_ALPHA, LORA_DROPOUT),
    "dora": PeftFamilySpec("dora", "DoRA", list(DORA_TARGET_MODULES), DORA_R, DORA_ALPHA, DORA_DROPOUT, use_dora=True),
    "adalora": PeftFamilySpec(
        "adalora",
        "AdaLoRA",
        list(ADALORA_TARGET_MODULES),
        ADALORA_INIT_R,
        ADALORA_ALPHA,
        ADALORA_DROPOUT,
        use_adalora=True,
    ),
    "qlora": PeftFamilySpec(
        "qlora",
        "QLoRA",
        list(QLORA_TARGET_MODULES),
        QLORA_R,
        QLORA_ALPHA,
        QLORA_DROPOUT,
        use_qlora=True,
    ),
}


def build_absa_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aspect_col_map = {
        "risk": "risk_sentiment",
        "trust": "trust_sentiment",
        "service": "service_sentiment",
    }

    for aspect, col in aspect_col_map.items():
        if col not in df.columns:
            continue
        subset = df[df[col].isin(LABEL2ID.keys())][["review_id", "review_text", col]].copy()
        subset["aspect"] = aspect
        subset.rename(columns={col: "label"}, inplace=True)
        rows.append(subset)

    combined = pd.concat(rows, ignore_index=True)
    combined["task_text"] = "[ASPECT=" + combined["aspect"] + "] " + combined["review_text"].astype(str)
    combined["label_id"] = combined["label"].map(LABEL2ID)
    combined = combined.dropna(subset=["task_text", "label_id"]).reset_index(drop=True)
    combined["label_id"] = combined["label_id"].astype(int)
    return combined


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


def _resolve_compute_dtype() -> torch.dtype:
    return getattr(torch, QLORA_COMPUTE_DTYPE, torch.float16)


def load_base_model(model_name: str, spec: PeftFamilySpec):
    common_kwargs = {
        "num_labels": 3,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "ignore_mismatched_sizes": True,
    }

    if spec.use_qlora:
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "QLoRA membutuhkan runtime GPU yang aktif dengan backend 4-bit yang tervalidasi."
            )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QLORA_QUANT_TYPE,
            bnb_4bit_use_double_quant=QLORA_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=_resolve_compute_dtype(),
        )
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                **common_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                "QLoRA gagal memuat model 4-bit. Pastikan bitsandbytes dan backend GPU yang dipakai kompatibel di droplet ini."
            ) from exc
        return prepare_model_for_kbit_training(model)

    return AutoModelForSequenceClassification.from_pretrained(model_name, **common_kwargs)


def build_peft_config(spec: PeftFamilySpec, total_steps: int):
    if spec.use_adalora:
        return AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=spec.target_modules,
            lora_alpha=spec.alpha,
            lora_dropout=spec.dropout,
            target_r=ADALORA_TARGET_R,
            init_r=ADALORA_INIT_R,
            tinit=ADALORA_TINIT,
            tfinal=ADALORA_TFINAL,
            deltaT=ADALORA_DELTA_T,
            beta1=ADALORA_BETA1,
            beta2=ADALORA_BETA2,
            orth_reg_weight=ADALORA_ORTH_REG_WEIGHT,
            total_step=max(int(total_steps), 1),
            bias="none",
        )

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=spec.r,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=spec.target_modules,
        bias="none",
        use_dora=spec.use_dora,
    )


def default_output_dir(family_name: str, filtered: bool) -> str:
    return str(MODELS_DIR / (f"retrained_{family_name}" if filtered else family_name))


def add_common_parser_args(parser, *, family_name: str, filtered: bool, default_lr: float):
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    parser.add_argument("--output_dir", default=default_output_dir(family_name, filtered))
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=TRAIN_MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--peft_r", type=int, default=FAMILY_SPECS[family_name].r)
    parser.add_argument("--peft_alpha", type=int, default=FAMILY_SPECS[family_name].alpha)
    parser.add_argument("--peft_dropout", type=float, default=FAMILY_SPECS[family_name].dropout)
    parser.add_argument("--adalora_init_r", type=int, default=ADALORA_INIT_R)
    parser.add_argument("--adalora_target_r", type=int, default=ADALORA_TARGET_R)
    parser.add_argument("--adalora_tinit", type=int, default=ADALORA_TINIT)
    parser.add_argument("--adalora_tfinal", type=int, default=ADALORA_TFINAL)
    parser.add_argument("--adalora_delta_t", type=int, default=ADALORA_DELTA_T)
    parser.add_argument("--adalora_beta1", type=float, default=ADALORA_BETA1)
    parser.add_argument("--adalora_beta2", type=float, default=ADALORA_BETA2)
    parser.add_argument("--adalora_orth_reg_weight", type=float, default=ADALORA_ORTH_REG_WEIGHT)
    if filtered:
        parser.add_argument("--clean_csv", default=str(DATA_PROCESSED / "noise" / "clean_data.csv"))
        parser.add_argument("--experiment_family", default=f"retrained_{family_name}")
        parser.add_argument("--uncertainty_source_model_id", default=None)
        parser.add_argument("--noise_summary_json", default=None)
        parser.add_argument("--mc_summary_json", default=None)
    else:
        parser.add_argument("--input_csv", default=str(DATA_PROCESSED / "dataset_absa_50k_v2_intersection.csv"))
        parser.add_argument("--experiment_family", default=family_name)


def build_variant_parser(family_name: str, *, filtered: bool = False):
    description = f"Train IndoBERT + {FAMILY_SPECS[family_name].display_name}"
    if filtered:
        description += " on clean uncertainty-filtered data"
    parser = __import__("argparse").ArgumentParser(description=description)
    default_lr = FULL_FINETUNE_DEFAULT_LR if family_name == "qlora" else PEFT_DEFAULT_LR
    add_common_parser_args(parser, family_name=family_name, filtered=filtered, default_lr=default_lr)
    return parser


def _prepare_data(args, *, filtered: bool) -> tuple[pd.DataFrame, dict]:
    if filtered:
        data = load_clean_data(args.clean_csv)
        uncertainty_meta = infer_uncertainty_source(args.clean_csv)
    else:
        df = pd.read_csv(args.input_csv)
        data = build_absa_rows(df)
        uncertainty_meta = {
            "uncertainty_enabled": False,
            "uncertainty_source_model_family": None,
            "uncertainty_source_run_name": None,
            "uncertainty_source_model_id": None,
        }

    if len(data) < 30:
        raise ValueError(f"Not enough labeled data: {len(data)} (need >= 30)")
    return data, uncertainty_meta


def run_peft_training(args, *, family_name: str, filtered: bool = False):
    spec = FAMILY_SPECS[family_name]
    set_seed(args.seed)
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data, uncertainty_meta = _prepare_data(args, filtered=filtered)
    train_df, val_df, test_df = review_level_split(
        data,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    split_manifest = write_split_manifest(output_path, train_df, val_df, test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenizer(train_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    val_enc = tokenizer(val_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)
    test_enc = tokenizer(test_df["task_text"].tolist(), truncation=True, padding=True, max_length=args.max_length)

    train_dataset = ABSADataset(train_enc, train_df["label_id"].tolist())
    val_dataset = ABSADataset(val_enc, val_df["label_id"].tolist())
    test_dataset = ABSADataset(test_enc, test_df["label_id"].tolist())

    total_steps = math.ceil(max(len(train_dataset), 1) / max(args.batch_size, 1)) * max(args.epochs, 1)
    base_model = load_base_model(args.model_name, spec)
    effective_rank = int(args.peft_r)
    effective_alpha = int(args.peft_alpha)
    effective_dropout = float(args.peft_dropout)

    runtime_spec = PeftFamilySpec(
        family_name=spec.family_name,
        display_name=spec.display_name,
        target_modules=list(spec.target_modules),
        r=effective_rank,
        alpha=effective_alpha,
        dropout=effective_dropout,
        use_dora=spec.use_dora,
        use_adalora=spec.use_adalora,
        use_qlora=spec.use_qlora,
    )

    if runtime_spec.use_adalora:
        if args.adalora_target_r > args.adalora_init_r:
            raise ValueError("AdaLoRA requires adalora_target_r <= adalora_init_r.")

    peft_config = build_peft_config(runtime_spec, total_steps)
    if runtime_spec.use_adalora:
        peft_config.init_r = int(args.adalora_init_r)
        peft_config.target_r = int(args.adalora_target_r)
        peft_config.tinit = int(args.adalora_tinit)
        peft_config.tfinal = int(args.adalora_tfinal)
        peft_config.deltaT = int(args.adalora_delta_t)
        peft_config.beta1 = float(args.adalora_beta1)
        peft_config.beta2 = float(args.adalora_beta2)
        peft_config.orth_reg_weight = float(args.adalora_orth_reg_weight)
    model = get_peft_model(base_model, peft_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct_trainable = trainable_params / total_params * 100

    training_kwargs = {
        "output_dir": str(output_path / "checkpoints"),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "report_to": "none",
        "save_total_limit": 2,
    }
    if spec.use_qlora:
        training_kwargs["gradient_checkpointing"] = True
        training_kwargs["fp16"] = torch.cuda.is_available()

    training_args = TrainingArguments(**training_kwargs)

    callbacks = [EpochTimingCallback()]
    if spec.use_adalora:
        callbacks.append(AdaLoraStepCallback())
    timing_callback = callbacks[0]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
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
        "uncertainty_enabled": bool(filtered),
        "uncertainty_source_model_family": uncertainty_meta["uncertainty_source_model_family"],
        "uncertainty_source_run_name": uncertainty_meta["uncertainty_source_run_name"],
        "uncertainty_source_model_id": (
            getattr(args, "uncertainty_source_model_id", None) or uncertainty_meta["uncertainty_source_model_id"]
        ),
        "noise_summary_json": getattr(args, "noise_summary_json", None),
        "mc_summary_json": getattr(args, "mc_summary_json", None),
        "peft_variant": family_name,
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
        "n_total_clean": len(data) if filtered else None,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(pct_trainable, 2),
        "peft_rank": runtime_spec.r,
        "peft_alpha": runtime_spec.alpha,
        "peft_dropout": runtime_spec.dropout,
        "target_modules": runtime_spec.target_modules,
        "uses_dora": runtime_spec.use_dora,
        "uses_adalora": runtime_spec.use_adalora,
        "uses_qlora": runtime_spec.use_qlora,
        "train_start_timestamp": timing_callback.train_start_timestamp,
        "train_end_timestamp": timing_callback.train_end_timestamp,
        "training_time_seconds": round(train_time, 2),
        "label_distribution": data["label"].value_counts().to_dict(),
        "aspect_distribution": data["aspect"].value_counts().to_dict(),
        "split_manifest": split_manifest,
    }
    if runtime_spec.use_adalora:
        metrics.update(
            {
                "adalora_init_r": int(args.adalora_init_r),
                "adalora_target_r": int(args.adalora_target_r),
                "adalora_tinit": int(args.adalora_tinit),
                "adalora_tfinal": int(args.adalora_tfinal),
                "adalora_delta_t": int(args.adalora_delta_t),
                "adalora_beta1": float(args.adalora_beta1),
                "adalora_beta2": float(args.adalora_beta2),
                "adalora_orth_reg_weight": float(args.adalora_orth_reg_weight),
                "adalora_total_step": total_steps,
            }
        )
    if runtime_spec.use_qlora:
        metrics.update(
            {
                "qlora_quant_type": QLORA_QUANT_TYPE,
                "qlora_double_quant": QLORA_DOUBLE_QUANT,
                "qlora_compute_dtype": QLORA_COMPUTE_DTYPE,
            }
        )

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

    label_col = "weak_label" if filtered else "label"
    preds_df = test_df[["review_id", "aspect", "review_text", label_col]].copy()
    preds_df["pred_label"] = [ID2LABEL[int(x)] for x in y_pred]
    preds_df["is_error"] = preds_df[label_col] != preds_df["pred_label"]
    preds_df.to_csv(output_path / "test_predictions.csv", index=False)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[{spec.display_name.upper()}] Training complete in {train_time:.1f}s")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
