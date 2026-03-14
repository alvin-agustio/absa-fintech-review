# DigitalOcean Training Runbook

Panduan ini untuk mengulang workflow training skripsi di GPU Droplet DigitalOcean dari nol sampai aman diakhiri.

## Kapan Aman Destroy Droplet

Droplet **boleh di-destroy** kalau semua ini sudah selesai:

- artefak hasil sudah diarsipkan di folder `exports/`
- file arsip penting sudah berhasil dipindahkan ke lokal
- arsip lokal bisa dibuka tanpa error
- model final terbaik sudah ikut tersimpan
- `git push` untuk perubahan kode yang penting sudah selesai

Checklist minimum sebelum destroy:

- `skripsi_eval_core.tar.gz`
- `skripsi_experiment_reports.tar.gz`
- `skripsi_best_model_retrained_lora_epoch8.tar.gz`
- `skripsi_metadata_bundle_*.tar.gz`

Jika ingin backup lebih lengkap:

- `skripsi_post_training_all_no_checkpoints_*.tar.gz`

## 1. Siapkan Droplet

Gunakan image Linux yang cocok dengan GPU yang dipakai.

- Untuk NVIDIA: image yang siap NVIDIA/CUDA
- Untuk AMD MI300X: image yang siap ROCm

Install tool dasar:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip tmux
```

## 2. Clone Repo

```bash
cd ~
git clone https://github.com/alvin-agustio/fintech-review-absa skripsi
cd skripsi
```

## 3. Buat Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install PyTorch Sesuai GPU

### AMD ROCm

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch torchvision torchaudio
```

### NVIDIA CUDA

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

## 5. Install Dependency Proyek

```bash
pip install -r requirements.txt
pip install peft streamlit plotly
```

## 6. Verifikasi GPU

### AMD

```bash
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
```

### NVIDIA

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## 7. Jalankan Eksperimen Utama

### Training baseline dan LoRA

```bash
python train_baseline.py
python train_lora.py
```

### Sweep epoch 3, 5, 8

Jika memakai PowerShell di Linux:

```bash
sed -i 's#\.\\.venv\\Scripts\\python\.exe#./.venv/bin/python#' scripts/run_training_experiments.ps1
pwsh ./scripts/run_training_experiments.ps1
```

## 8. Uncertainty dan Noise Filtering

Gunakan model weak-label terbaik sebagai dasar.

Contoh:

```bash
python predict_mc_dropout.py --input_csv data/processed/dataset_absa_50k_v2_intersection.csv --model_dir models/baseline/epoch_5/model --num_mc 30 --batch_size 64
python detect_label_noise.py
```

## 9. Retrain pada Clean Subset

### Full fine-tuning

```bash
python retrain_filtered.py --epochs 3 --output_dir models/retrained/epoch_3
python retrain_filtered.py --epochs 5 --output_dir models/retrained/epoch_5
python retrain_filtered.py --epochs 8 --output_dir models/retrained/epoch_8
```

### LoRA pada clean subset

```bash
python train_lora_filtered.py --epochs 3 --output_dir models/retrained_lora/epoch_3
python train_lora_filtered.py --epochs 5 --output_dir models/retrained_lora/epoch_5
python train_lora_filtered.py --epochs 8 --output_dir models/retrained_lora/epoch_8
```

## 10. Evaluasi Final

```bash
python evaluate.py
```

File penting hasil evaluasi:

- `data/processed/evaluation/evaluation_summary.json`
- `data/processed/evaluation/evaluation_detailed.json`
- `data/processed/evaluation/epoch_comparison_summary.csv`
- `data/processed/evaluation/epoch_comparison_wide.csv`

## 11. Arsipkan Hasil Penting

### Inti evaluasi

```bash
tar -czf exports/skripsi_eval_core.tar.gz \
  data/processed/evaluation \
  data/processed/uncertainty \
  data/processed/noise
```

### Laporan eksperimen ringkas

```bash
tar -czf exports/skripsi_experiment_reports.tar.gz \
  models/baseline/epoch_3/metrics.json \
  models/baseline/epoch_3/classification_report.txt \
  models/baseline/epoch_3/test_predictions.csv \
  models/baseline/epoch_5/metrics.json \
  models/baseline/epoch_5/classification_report.txt \
  models/baseline/epoch_5/test_predictions.csv \
  models/baseline/epoch_8/metrics.json \
  models/baseline/epoch_8/classification_report.txt \
  models/baseline/epoch_8/test_predictions.csv \
  models/lora/epoch_3/metrics.json \
  models/lora/epoch_3/classification_report.txt \
  models/lora/epoch_3/test_predictions.csv \
  models/lora/epoch_5/metrics.json \
  models/lora/epoch_5/classification_report.txt \
  models/lora/epoch_5/test_predictions.csv \
  models/lora/epoch_8/metrics.json \
  models/lora/epoch_8/classification_report.txt \
  models/lora/epoch_8/test_predictions.csv \
  models/retrained/epoch_3/metrics.json \
  models/retrained/epoch_3/classification_report.txt \
  models/retrained/epoch_3/test_predictions.csv \
  models/retrained/epoch_5/metrics.json \
  models/retrained/epoch_5/classification_report.txt \
  models/retrained/epoch_5/test_predictions.csv \
  models/retrained/epoch_8/metrics.json \
  models/retrained/epoch_8/classification_report.txt \
  models/retrained/epoch_8/test_predictions.csv \
  models/retrained_lora/epoch_3/metrics.json \
  models/retrained_lora/epoch_3/classification_report.txt \
  models/retrained_lora/epoch_3/test_predictions.csv \
  models/retrained_lora/epoch_5/metrics.json \
  models/retrained_lora/epoch_5/classification_report.txt \
  models/retrained_lora/epoch_5/test_predictions.csv \
  models/retrained_lora/epoch_8/metrics.json \
  models/retrained_lora/epoch_8/classification_report.txt \
  models/retrained_lora/epoch_8/test_predictions.csv
```

### Model final terbaik

```bash
tar -czf exports/skripsi_best_model_retrained_lora_epoch8.tar.gz \
  models/retrained_lora/epoch_8/model
```

### Paket lengkap pasca-training tanpa checkpoint

```bash
STAMP=$(date +%F_%H%M%S)
tar -czf exports/skripsi_post_training_all_no_checkpoints_${STAMP}.tar.gz \
  --exclude='*/checkpoints/*' \
  data/processed/uncertainty \
  data/processed/noise \
  data/processed/evaluation \
  models/baseline/epoch_3 \
  models/baseline/epoch_5 \
  models/baseline/epoch_8 \
  models/lora/epoch_3 \
  models/lora/epoch_5 \
  models/lora/epoch_8 \
  models/retrained/epoch_3 \
  models/retrained/epoch_5 \
  models/retrained/epoch_8 \
  models/retrained_lora/epoch_3 \
  models/retrained_lora/epoch_5 \
  models/retrained_lora/epoch_8
```

## 12. Simpan Metadata Reproducibility

```bash
git rev-parse HEAD > exports/git_commit.txt
git status --short > exports/git_status.txt
pip freeze > exports/pip_freeze.txt
python -c "import torch; print(torch.__version__); print(torch.version.hip or torch.version.cuda); print(torch.cuda.is_available())" > exports/torch_env.txt
tar -czf exports/skripsi_metadata_bundle.tar.gz \
  exports/git_commit.txt \
  exports/git_status.txt \
  exports/pip_freeze.txt \
  exports/torch_env.txt
```

## 13. Download ke Lokal

Pindahkan file arsip penting ke lokal.

Contoh:

```powershell
scp -i $HOME\.ssh\id_ed25519_nopass root@YOUR_DROPLET_IP:/root/skripsi/exports/skripsi_eval_core.tar.gz C:\Users\alvin\Downloads\
```

## 14. Verifikasi Lokal

Pastikan file hasil download bisa dibuka:

```powershell
tar -tf C:\Users\alvin\Downloads\skripsi_eval_core.tar.gz
```

Lakukan juga untuk arsip penting lain.

## 15. Baru Destroy Droplet

Kalau semua arsip penting sudah:

- selesai didownload
- lolos verifikasi
- aman di lokal / cloud backup

maka droplet aman untuk di-destroy.
