#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm6.3}"

echo "[1/8] Checking ROCm tools..."
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | head -n 40 || true
else
  echo "rocminfo not found."
  echo "Use a DigitalOcean AMD/ROCm-ready image first."
  exit 1
fi

echo "[2/8] Checking AMD SMI..."
if command -v amd-smi >/dev/null 2>&1; then
  amd-smi static --gpu 0 || true
else
  echo "amd-smi not found. Continuing, but ROCm visibility should be checked manually."
fi

echo "[3/8] Checking Python..."
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}"
  exit 1
fi

echo "[4/8] Creating virtual environment at ${VENV_DIR}..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[5/8] Upgrading pip tooling..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[6/8] Installing ROCm-enabled PyTorch from ${TORCH_INDEX_URL}..."
"${VENV_DIR}/bin/python" -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio

echo "[7/8] Installing project dependencies..."
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install streamlit plotly

echo "[8/8] Running quick ROCm validation..."
"${VENV_DIR}/bin/python" - <<'PY'
import torch
print("torch_version =", torch.__version__)
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.version.hip =", torch.version.hip)
if torch.cuda.is_available():
    print("device_name =", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("device_memory_gb =", round(props.total_memory / (1024 ** 3), 2))
PY

echo
echo "ROCm setup complete."
echo "Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Recommended next check:"
echo "  python scripts/check_qlora_rocm_smoke.py"
