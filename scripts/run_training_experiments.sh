#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
INCLUDE_QLORA="${INCLUDE_QLORA:-0}"

INPUT_CSV="data/processed/dataset_absa_50k_v2_intersection.csv"
MODEL_NAME="indobenchmark/indobert-base-p1"
MAX_LENGTH=128
TEST_SIZE=0.2
VAL_SIZE=0.1
SEED=42
MAX_EPOCH=15
BATCH_SIZE=8

BASELINE_LR="2e-5"
PEFT_LR="2e-4"
QLORA_LR="2e-5"

run_train_job() {
  local label="$1"
  local script_path="$2"
  local output_dir="$3"
  local lr="$4"
  local experiment_family="$5"

  echo "[$label] Running -> $output_dir"
  "${PYTHON_BIN}" "${script_path}" \
    --input_csv "${INPUT_CSV}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${output_dir}" \
    --max_length "${MAX_LENGTH}" \
    --test_size "${TEST_SIZE}" \
    --val_size "${VAL_SIZE}" \
    --epochs "${MAX_EPOCH}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${lr}" \
    --seed "${SEED}" \
    --experiment_family "${experiment_family}"
}

run_retrain_job() {
  local label="$1"
  local script_path="$2"
  local clean_csv="$3"
  local output_dir="$4"
  local lr="$5"
  local experiment_family="$6"
  local source_model_id="$7"
  local source_family="$8"

  if [[ ! -f "${clean_csv}" ]]; then
    echo "[RUNNER] Skip ${label} because clean data was not found at ${clean_csv}"
    return
  fi

  echo "[$label] Running -> $output_dir"
  "${PYTHON_BIN}" "${script_path}" \
    --clean_csv "${clean_csv}" \
    --output_dir "${output_dir}" \
    --max_length "${MAX_LENGTH}" \
    --test_size "${TEST_SIZE}" \
    --val_size "${VAL_SIZE}" \
    --epochs "${MAX_EPOCH}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${lr}" \
    --seed "${SEED}" \
    --experiment_family "${experiment_family}" \
    --uncertainty_source_model_id "${source_model_id}" \
    --noise_summary_json "data/processed/noise/${source_family}/epoch_15/noise_summary.json" \
    --mc_summary_json "data/processed/uncertainty/${source_family}/epoch_15/mc_summary.json"
}

run_train_job "BASELINE" "src/training/train_baseline.py" "models/baseline/epoch_15" "${BASELINE_LR}" "baseline"
run_train_job "LORA" "src/training/train_lora.py" "models/lora/epoch_15" "${PEFT_LR}" "lora"
run_train_job "DORA" "src/training/train_dora.py" "models/dora/epoch_15" "${PEFT_LR}" "dora"
run_train_job "ADALORA" "src/training/train_adalora.py" "models/adalora/epoch_15" "${PEFT_LR}" "adalora"

if [[ "${INCLUDE_QLORA}" == "1" ]]; then
  run_train_job "QLORA" "src/training/train_qlora.py" "models/qlora/epoch_15" "${QLORA_LR}" "qlora"
fi

run_retrain_job "RETRAIN" "src/training/retrain_filtered.py" \
  "data/processed/noise/baseline/epoch_15/clean_data.csv" "models/retrained/epoch_15" \
  "${BASELINE_LR}" "retrained" "baseline:epoch_15" "baseline"

run_retrain_job "RETRAIN-LORA" "src/training/train_lora_filtered.py" \
  "data/processed/noise/lora/epoch_15/clean_data.csv" "models/retrained_lora/epoch_15" \
  "${PEFT_LR}" "retrained_lora" "lora:epoch_15" "lora"

run_retrain_job "RETRAIN-DORA" "src/training/train_dora_filtered.py" \
  "data/processed/noise/dora/epoch_15/clean_data.csv" "models/retrained_dora/epoch_15" \
  "${PEFT_LR}" "retrained_dora" "dora:epoch_15" "dora"

run_retrain_job "RETRAIN-ADALORA" "src/training/train_adalora_filtered.py" \
  "data/processed/noise/adalora/epoch_15/clean_data.csv" "models/retrained_adalora/epoch_15" \
  "${PEFT_LR}" "retrained_adalora" "adalora:epoch_15" "adalora"

if [[ "${INCLUDE_QLORA}" == "1" ]]; then
  run_retrain_job "RETRAIN-QLORA" "src/training/train_qlora_filtered.py" \
    "data/processed/noise/qlora/epoch_15/clean_data.csv" "models/retrained_qlora/epoch_15" \
    "${QLORA_LR}" "retrained_qlora" "qlora:epoch_15" "qlora"
fi

echo "[RUNNER] All requested training runs up to epoch ${MAX_EPOCH} completed successfully."
