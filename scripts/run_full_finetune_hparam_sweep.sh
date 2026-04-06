#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
FAMILY="${FAMILY:-baseline}"
INPUT_CSV="${INPUT_CSV:-data/processed/dataset_absa_50k_v2_intersection.csv}"
CLEAN_CSV="${CLEAN_CSV:-data/processed/noise/baseline/epoch_15/clean_data.csv}"
MODEL_NAME="${MODEL_NAME:-indobenchmark/indobert-base-p1}"
MAX_LENGTH="${MAX_LENGTH:-128}"
TEST_SIZE="${TEST_SIZE:-0.2}"
VAL_SIZE="${VAL_SIZE:-0.1}"
SEED="${SEED:-42}"

CFG_JSON="$("${PYTHON_BIN}" scripts/export_hparam_sweep_config.py)"
mapfile -t LR_CANDIDATES < <(CFG_JSON="${CFG_JSON}" python - <<'PY'
import json, os
cfg = json.loads(os.environ["CFG_JSON"])
for item in cfg["full_finetune_lr_candidates"]:
    print(item)
PY
)
EPOCHS="$(CFG_JSON="${CFG_JSON}" python - <<'PY'
import json, os
print(json.loads(os.environ["CFG_JSON"])["train_max_epochs"])
PY
)"
BATCH_SIZE="$(CFG_JSON="${CFG_JSON}" python - <<'PY'
import json, os
print(json.loads(os.environ["CFG_JSON"])["train_batch_size"])
PY
)"

ROOT_OUTPUT="experiments/hparam_sweeps/${FAMILY}"
mkdir -p "${ROOT_OUTPUT}"
SUMMARY_CSV="${ROOT_OUTPUT}/sweep_summary.csv"
echo "family,learning_rate,best_epoch,best_validation_f1_macro,best_validation_accuracy,best_validation_precision_macro,best_validation_recall_macro,training_time_seconds,output_dir" > "${SUMMARY_CSV}"

for lr in "${LR_CANDIDATES[@]}"; do
  lr_token="${lr//./p}"
  output_dir="${ROOT_OUTPUT}/lr_${lr_token}"

  if [[ "${FAMILY}" == "baseline" ]]; then
    "${PYTHON_BIN}" src/training/train_baseline.py \
      --input_csv "${INPUT_CSV}" \
      --model_name "${MODEL_NAME}" \
      --output_dir "${output_dir}" \
      --max_length "${MAX_LENGTH}" \
      --test_size "${TEST_SIZE}" \
      --val_size "${VAL_SIZE}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${lr}" \
      --seed "${SEED}" \
      --experiment_family "baseline_tuning"
  else
    if [[ ! -f "${CLEAN_CSV}" ]]; then
      echo "Clean CSV not found for retrained sweep: ${CLEAN_CSV}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" src/training/retrain_filtered.py \
      --clean_csv "${CLEAN_CSV}" \
      --model_name "${MODEL_NAME}" \
      --output_dir "${output_dir}" \
      --max_length "${MAX_LENGTH}" \
      --test_size "${TEST_SIZE}" \
      --val_size "${VAL_SIZE}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${lr}" \
      --seed "${SEED}" \
      --experiment_family "retrained_tuning"
  fi

  METRICS_PATH="${output_dir}/metrics.json"
  METRICS_PATH="${METRICS_PATH}" LR="${lr}" FAMILY="${FAMILY}" OUTPUT_DIR="${output_dir}" python - <<'PY' >> "${SUMMARY_CSV}"
import json, os
from pathlib import Path
metrics = json.loads(Path(os.environ["METRICS_PATH"]).read_text(encoding="utf-8"))
row = [
    os.environ["FAMILY"],
    os.environ["LR"],
    metrics.get("best_epoch"),
    metrics.get("best_validation_f1_macro"),
    metrics.get("best_validation_accuracy"),
    metrics.get("best_validation_precision_macro"),
    metrics.get("best_validation_recall_macro"),
    metrics.get("training_time_seconds"),
    os.environ["OUTPUT_DIR"],
]
print(",".join("" if value is None else str(value) for value in row))
PY
done

echo "[SWEEP] Summary written to ${SUMMARY_CSV}"
