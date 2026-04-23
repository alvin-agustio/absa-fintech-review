#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPORT_DIR="${ROOT_DIR}/exports"
AUDIT_DIR="${EXPORT_DIR}/checkpoint_audit_light"
BUNDLE_NAME="${1:-skripsi_models_final_plus_checkpoint_audit_light.tar.gz}"
BUNDLE_PATH="${EXPORT_DIR}/${BUNDLE_NAME}"

mkdir -p "$EXPORT_DIR" "$AUDIT_DIR"

FINAL_FAMILIES=(
  baseline
  lora
  dora
  adalora
  qlora
  retrained
  retrained_lora
  retrained_dora
  retrained_adalora
  retrained_qlora
)

MODEL_LIST_FILE="${AUDIT_DIR}/final_model_dirs.txt"
CHECKPOINT_LIST_FILE="${AUDIT_DIR}/checkpoint_light_file_list.txt"
INVENTORY_FILE="${AUDIT_DIR}/checkpoint_file_inventory.txt"
SUMMARY_FILE="${AUDIT_DIR}/bundle_summary.txt"

: > "$MODEL_LIST_FILE"
: > "$CHECKPOINT_LIST_FILE"
: > "$INVENTORY_FILE"

for family in "${FINAL_FAMILIES[@]}"; do
  model_dir="models/${family}/epoch_15/model"
  if [[ -d "$model_dir" ]]; then
    printf '%s\n' "$model_dir" >> "$MODEL_LIST_FILE"
  fi
done

find models -path "*/checkpoints/checkpoint-*" -type f -printf "%P|%s bytes\n" | sort > "$INVENTORY_FILE"

find models -path "*/checkpoints/checkpoint-*" \
  \( -name "trainer_state.json" \
  -o -name "training_args.bin" \
  -o -name "config.json" \
  -o -name "adapter_config.json" \
  -o -name "README.md" \
  -o -name "rng_state.pth" \
  -o -name "scheduler.pt" \) \
  -type f | sort > "$CHECKPOINT_LIST_FILE"

MODEL_COUNT="$(wc -l < "$MODEL_LIST_FILE" | tr -d ' ')"
CHECKPOINT_FILE_COUNT="$(wc -l < "$CHECKPOINT_LIST_FILE" | tr -d ' ')"

cat > "$SUMMARY_FILE" <<EOF
Bundle: ${BUNDLE_NAME}
Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Root: ${ROOT_DIR}
Final model directories included: ${MODEL_COUNT}
Light checkpoint files included: ${CHECKPOINT_FILE_COUNT}

Included final model directories:
$(cat "$MODEL_LIST_FILE")
EOF

tar -czf "$BUNDLE_PATH" \
  -T "$MODEL_LIST_FILE" \
  -T "$CHECKPOINT_LIST_FILE" \
  "$INVENTORY_FILE" \
  "$CHECKPOINT_LIST_FILE" \
  "$MODEL_LIST_FILE" \
  "$SUMMARY_FILE"

echo "[OK] Bundle created: ${BUNDLE_PATH}"
ls -lh "$BUNDLE_PATH"
