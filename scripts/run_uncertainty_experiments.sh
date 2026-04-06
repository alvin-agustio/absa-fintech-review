#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
INCLUDE_QLORA="${INCLUDE_QLORA:-0}"

INPUT_CSV="data/processed/dataset_absa_50k_v2_intersection.csv"
MAX_LENGTH=128
NUM_MC=30
BATCH_SIZE=16
UNCERTAINTY_COL="uncertainty_entropy"
HIGH_UNCERTAINTY_QUANTILE=0.8

run_family_uncertainty() {
  local family="$1"
  local model_dir="$2"

  if [[ ! -d "${model_dir}" ]]; then
    echo "[UNCERTAINTY] Skip ${family} because model_dir not found: ${model_dir}"
    return
  fi

  echo "[UNCERTAINTY] MC Dropout -> ${family} / epoch_15"
  "${PYTHON_BIN}" src/evaluation/predict_mc_dropout.py \
    --input_csv "${INPUT_CSV}" \
    --model_dir "${model_dir}" \
    --model_family "${family}" \
    --run_name "epoch_15" \
    --output_dir "data/processed/uncertainty" \
    --num_mc "${NUM_MC}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}"

  echo "[UNCERTAINTY] Noise detection -> ${family} / epoch_15"
  "${PYTHON_BIN}" src/evaluation/detect_label_noise.py \
    --input_dir "data/processed/uncertainty/${family}/epoch_15" \
    --model_family "${family}" \
    --run_name "epoch_15" \
    --output_dir "data/processed/noise" \
    --uncertainty_col "${UNCERTAINTY_COL}" \
    --high_uncertainty_quantile "${HIGH_UNCERTAINTY_QUANTILE}" \
    --threshold_scope global
}

run_family_uncertainty "baseline" "models/baseline/epoch_15/model"
run_family_uncertainty "lora" "models/lora/epoch_15/model"
run_family_uncertainty "dora" "models/dora/epoch_15/model"
run_family_uncertainty "adalora" "models/adalora/epoch_15/model"

if [[ "${INCLUDE_QLORA}" == "1" ]]; then
  run_family_uncertainty "qlora" "models/qlora/epoch_15/model"
fi

echo "[UNCERTAINTY] Family-aware uncertainty runs completed."
