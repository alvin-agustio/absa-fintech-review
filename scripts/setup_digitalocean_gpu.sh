#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "[1/6] Checking NVIDIA GPU availability..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found."
  echo "Use a DigitalOcean AI/ML-ready NVIDIA image or install NVIDIA drivers first."
  exit 1
fi

echo "[2/6] Checking Python..."
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python not found: ${PYTHON_BIN}"
  echo "Install Python 3.10+ first, then rerun this script."
  exit 1
fi

echo "[3/6] Creating virtual environment at ${VENV_DIR}..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[4/6] Upgrading pip tooling..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[5/6] Installing CUDA-enabled PyTorch from ${TORCH_INDEX_URL}..."
"${VENV_DIR}/bin/python" -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio

echo "[6/6] Installing project dependencies..."
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"

echo
echo "Setup complete."
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo
echo "Quick GPU check:"
echo "  python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')\""
echo
echo "Start LoRA training:"
echo "  python train_lora.py"
echo
echo "Start Streamlit dashboard:"
echo "  streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
