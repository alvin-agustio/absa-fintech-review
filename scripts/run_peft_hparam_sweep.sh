#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
FAMILY="${FAMILY:-lora}"
FILTERED="${FILTERED:-0}"
INPUT_CSV="${INPUT_CSV:-data/processed/dataset_absa_50k_v2_intersection.csv}"
CLEAN_CSV="${CLEAN_CSV:-}"
MODEL_NAME="${MODEL_NAME:-indobenchmark/indobert-base-p1}"
MAX_LENGTH="${MAX_LENGTH:-128}"
TEST_SIZE="${TEST_SIZE:-0.2}"
VAL_SIZE="${VAL_SIZE:-0.1}"
SEED="${SEED:-42}"

CFG_JSON="$("${PYTHON_BIN}" scripts/export_hparam_sweep_config.py)"
mapfile -t LR_CANDIDATES < <(CFG_JSON="${CFG_JSON}" python - <<'PY'
import json, os
for item in json.loads(os.environ["CFG_JSON"])["peft_lr_candidates"]:
    print(item)
PY
)
mapfile -t DROPOUT_CANDIDATES < <(CFG_JSON="${CFG_JSON}" python - <<'PY'
import json, os
for item in json.loads(os.environ["CFG_JSON"])["peft_dropout_candidates"]:
    print(item)
PY
)
if [[ "${FAMILY}" == "adalora" ]]; then
  mapfile -t RANK_CANDIDATES < <(CFG_JSON="${CFG_JSON}" python - <<'PY'
import json, os
for item in json.loads(os.environ["CFG_JSON"])["adalora_init_r_candidates"]:
    print(item)
PY
)
else
  mapfile -t RANK_CANDIDATES < <(CFG_JSON="${CFG_JSON}" python - <<'PY'
import json, os
for item in json.loads(os.environ["CFG_JSON"])["peft_r_candidates"]:
    print(item)
PY
)
fi
ADALORA_TARGET_R="$(CFG_JSON="${CFG_JSON}" python - <<'PY'
import json, os
print(json.loads(os.environ["CFG_JSON"])["adalora_target_r_default"])
PY
)"
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

SCOPE="standard"
if [[ "${FILTERED}" == "1" ]]; then
  SCOPE="filtered"
  if [[ -z "${CLEAN_CSV}" || ! -f "${CLEAN_CSV}" ]]; then
    echo "Set CLEAN_CSV to an existing clean subset when FILTERED=1" >&2
    exit 1
  fi
fi

SCRIPT_PATH="src/training/train_${FAMILY}.py"
if [[ "${FILTERED}" == "1" ]]; then
  SCRIPT_PATH="src/training/train_${FAMILY}_filtered.py"
fi

ROOT_OUTPUT="experiments/hparam_sweeps/${FAMILY}/${SCOPE}"
mkdir -p "${ROOT_OUTPUT}"
SUMMARY_CSV="${ROOT_OUTPUT}/sweep_summary.csv"
echo "family,scope,learning_rate,rank,dropout,best_epoch,best_validation_f1_macro,best_validation_accuracy,best_validation_precision_macro,best_validation_recall_macro,training_time_seconds,output_dir" > "${SUMMARY_CSV}"

for lr in "${LR_CANDIDATES[@]}"; do
  for rank in "${RANK_CANDIDATES[@]}"; do
    for dropout in "${DROPOUT_CANDIDATES[@]}"; do
      lr_token="${lr//./p}"
      dropout_token="${dropout//./p}"
      output_dir="${ROOT_OUTPUT}/lr_${lr_token}/r_${rank}/dropout_${dropout_token}"

      args=(
        "${SCRIPT_PATH}"
        --model_name "${MODEL_NAME}"
        --output_dir "${output_dir}"
        --max_length "${MAX_LENGTH}"
        --test_size "${TEST_SIZE}"
        --val_size "${VAL_SIZE}"
        --epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --lr "${lr}"
        --seed "${SEED}"
        --peft_alpha 32
        --peft_dropout "${dropout}"
        --experiment_family "${FAMILY}_tuning"
      )

      if [[ "${FILTERED}" == "1" ]]; then
        args+=( --clean_csv "${CLEAN_CSV}" )
      else
        args+=( --input_csv "${INPUT_CSV}" )
      fi

      if [[ "${FAMILY}" == "adalora" ]]; then
        target_rank="${ADALORA_TARGET_R}"
        if (( rank < target_rank )); then
          target_rank=$(( rank / 2 ))
          if (( target_rank < 4 )); then target_rank=4; fi
        fi
        args+=( --peft_r "${rank}" --adalora_init_r "${rank}" --adalora_target_r "${target_rank}" )
      else
        args+=( --peft_r "${rank}" )
      fi

      "${PYTHON_BIN}" "${args[@]}"

      METRICS_PATH="${output_dir}/metrics.json"
      METRICS_PATH="${METRICS_PATH}" LR="${lr}" RANK="${rank}" DROPOUT="${dropout}" FAMILY="${FAMILY}" SCOPE="${SCOPE}" OUTPUT_DIR="${output_dir}" python - <<'PY' >> "${SUMMARY_CSV}"
import json, os
from pathlib import Path
metrics = json.loads(Path(os.environ["METRICS_PATH"]).read_text(encoding="utf-8"))
row = [
    os.environ["FAMILY"],
    os.environ["SCOPE"],
    os.environ["LR"],
    os.environ["RANK"],
    os.environ["DROPOUT"],
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
  done
done

echo "[SWEEP] Summary written to ${SUMMARY_CSV}"
