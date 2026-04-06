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

Atau langsung pakai setup ROCm native untuk MI300X:

```bash
chmod +x scripts/setup_digitalocean_rocm.sh
./scripts/setup_digitalocean_rocm.sh
source .venv/bin/activate
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

### Training baseline dan keluarga PEFT utama

```bash
python src/training/train_baseline.py
python src/training/train_lora.py
python src/training/train_dora.py
python src/training/train_adalora.py
```

Untuk `QLoRA`, jalankan hanya setelah smoke test backend 4-bit lolos:

```bash
python scripts/check_qlora_rocm_smoke.py
python src/training/train_qlora.py
```

### Run penuh sampai 15 epoch

Gunakan runner shell-native:

```bash
bash scripts/run_training_experiments.sh
```

Jika ingin ikut menjalankan `QLoRA`:

```bash
INCLUDE_QLORA=1 bash scripts/run_training_experiments.sh
```

PowerShell Linux tetap bisa dipakai jika memang diperlukan:

```bash
sed -i 's#\.\\.venv\\Scripts\\python\.exe#./.venv/bin/python#' scripts/run_training_experiments.ps1
pwsh ./scripts/run_training_experiments.ps1
```

## 8. Uncertainty dan Noise Filtering

Gunakan jalur family-aware, bukan satu `clean_data.csv` global.

```bash
bash scripts/run_uncertainty_experiments.sh
```

Script ini menulis artefak seperti:

- `data/processed/uncertainty/baseline/epoch_15/mc_predictions.csv`
- `data/processed/uncertainty/lora/epoch_15/mc_predictions.csv`
- `data/processed/uncertainty/dora/epoch_15/mc_predictions.csv`
- `data/processed/uncertainty/adalora/epoch_15/mc_predictions.csv`
- `data/processed/noise/baseline/epoch_15/clean_data.csv`
- `data/processed/noise/lora/epoch_15/clean_data.csv`
- `data/processed/noise/dora/epoch_15/clean_data.csv`
- `data/processed/noise/adalora/epoch_15/clean_data.csv`

Jika ingin ikut menjalankan `QLoRA`:

```bash
INCLUDE_QLORA=1 bash scripts/run_uncertainty_experiments.sh
```

## 9. Retrain pada Clean Subset

### Full fine-tuning

```bash
bash scripts/run_uncertainty_retraining.sh
```

### Keluarga PEFT pada clean subset

Script di atas akan menjalankan retraining full fine-tuning dan keluarga PEFT pada clean subset family masing-masing.

Untuk ikut menjalankan `retrained_qlora`:

```bash
INCLUDE_QLORA=1 bash scripts/run_uncertainty_retraining.sh
```

## 10. Evaluasi Final

```bash
python src/evaluation/evaluate.py
python src/evaluation/evaluate_gold_subset.py
```

File penting hasil evaluasi:

- `data/processed/evaluation/evaluation_summary.json`
- `data/processed/evaluation/evaluation_detailed.json`
- `data/processed/evaluation/epoch_comparison_summary.csv`
- `data/processed/evaluation/epoch_comparison_wide.csv`
- `data/processed/evaluation/model_comparison_table.csv`
- `data/processed/evaluation/comparison_group_best.csv`
- `data/processed/diamond/evaluation/gold_evaluation_overview.csv`
- `data/processed/diamond/evaluation/gold_evaluation_group_best.csv`

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
  models/baseline/epoch_15/metrics.json \
  models/baseline/epoch_15/classification_report.txt \
  models/baseline/epoch_15/test_predictions.csv \
  models/lora/epoch_15/metrics.json \
  models/lora/epoch_15/classification_report.txt \
  models/lora/epoch_15/test_predictions.csv \
  models/dora/epoch_15/metrics.json \
  models/dora/epoch_15/classification_report.txt \
  models/dora/epoch_15/test_predictions.csv \
  models/adalora/epoch_15/metrics.json \
  models/adalora/epoch_15/classification_report.txt \
  models/adalora/epoch_15/test_predictions.csv \
  models/retrained/epoch_15/metrics.json \
  models/retrained/epoch_15/classification_report.txt \
  models/retrained/epoch_15/test_predictions.csv \
  models/retrained_lora/epoch_15/metrics.json \
  models/retrained_lora/epoch_15/classification_report.txt \
  models/retrained_lora/epoch_15/test_predictions.csv \
  models/retrained_dora/epoch_15/metrics.json \
  models/retrained_dora/epoch_15/classification_report.txt \
  models/retrained_dora/epoch_15/test_predictions.csv \
  models/retrained_adalora/epoch_15/metrics.json \
  models/retrained_adalora/epoch_15/classification_report.txt \
  models/retrained_adalora/epoch_15/test_predictions.csv
```

### Model final terbaik

```bash
tar -czf exports/skripsi_best_model.tar.gz \
  models/retrained_lora/epoch_15/model
```

### Paket lengkap pasca-training tanpa checkpoint

```bash
STAMP=$(date +%F_%H%M%S)
tar -czf exports/skripsi_post_training_all_no_checkpoints_${STAMP}.tar.gz \
  --exclude='*/checkpoints/*' \
  data/processed/uncertainty \
  data/processed/noise \
  data/processed/evaluation \
  models/baseline/epoch_15 \
  models/lora/epoch_15 \
  models/dora/epoch_15 \
  models/adalora/epoch_15 \
  models/retrained/epoch_15 \
  models/retrained_lora/epoch_15 \
  models/retrained_dora/epoch_15 \
  models/retrained_adalora/epoch_15
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
