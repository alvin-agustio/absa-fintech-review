# DigitalOcean Master Playbook

Dokumen ini adalah acuan utama untuk menjalankan eksperimen skripsi ini di DigitalOcean dari nol sampai selesai.

Tujuan dokumen ini:

- menjadi satu sumber kebenaran yang bisa dipanggil ulang nanti
- mengurangi kebingungan antar dokumen lama dan flow baru
- memastikan setup, training, uncertainty, evaluasi, dan arsip akhir tetap konsisten
- memberi checkpoint keputusan yang jelas sebelum lanjut ke tahap berikutnya

Dokumen ini ditulis dengan asumsi utama:

- target droplet utama adalah **AMD MI300X / ROCm**
- training utama memakai runner shell-native `.sh`
- eksperimen utama membandingkan:
  - baseline
  - LoRA
  - DoRA
  - AdaLoRA
  - QLoRA sebagai jalur opsional bertahap
- setiap family dibandingkan dalam dua kondisi:
  - without uncertainty
  - with uncertainty
- validasi akhir tetap dua lapis:
  - weak-label / LLM-labelled
  - human subset / gold subset

---

## 1. Dokumen Ini Menang Atas Apa

Kalau nanti ada beberapa markdown yang isinya terlihat mirip, pakai urutan prioritas ini:

1. dokumen ini
2. [DIGITALOCEAN_READY_MANIFEST.md](/c:/Users/alvin/Downloads/skripsi/docs/DIGITALOCEAN_READY_MANIFEST.md)
3. [DIGITALOCEAN_TRAINING_RUNBOOK.md](/c:/Users/alvin/Downloads/skripsi/docs/DIGITALOCEAN_TRAINING_RUNBOOK.md)
4. [DIGITALOCEAN_GPU_SETUP.md](/c:/Users/alvin/Downloads/skripsi/docs/DIGITALOCEAN_GPU_SETUP.md)

Catatan:

- `DIGITALOCEAN_GPU_SETUP.md` lebih lama dan masih kuat nuansa NVIDIA/CUDA
- untuk flow MI300X/ROCm terbaru, dokumen ini adalah acuan utama

---

## 2. Gambaran Besar Flow

Flow besarnya seperti ini:

1. siapkan droplet
2. clone repo
3. setup ROCm + virtual environment
4. verifikasi Python, torch, ROCm, dan GPU
5. jalankan smoke test QLoRA
6. jalankan tuning kecil jika dibutuhkan
7. jalankan training utama without uncertainty
8. jalankan uncertainty family-aware
9. jalankan retraining pada clean subset family masing-masing
10. jalankan evaluasi weak-label
11. jalankan evaluasi gold subset
12. arsipkan hasil penting
13. pindahkan hasil ke lokal
14. verifikasi hasil lokal
15. baru destroy droplet

---

## 3. Prinsip Operasional yang Wajib Diingat

### 3.1 Prinsip fairness

Yang dikunci untuk fairness:

- `max epoch = 15`
- `train batch size = 8`
- split `train/validation/test` harus konsisten
- pemilihan checkpoint terbaik harus berdasarkan validation, bukan test
- keputusan model final harus tetap dikonfirmasi di gold subset

### 3.2 Prinsip uncertainty

Uncertainty harus diperlakukan sebagai faktor eksperimen, bukan trik tambahan.

Artinya:

- baseline punya jalur uncertainty sendiri
- LoRA punya jalur uncertainty sendiri
- DoRA punya jalur uncertainty sendiri
- AdaLoRA punya jalur uncertainty sendiri
- QLoRA punya jalur uncertainty sendiri jika dijalankan

Tidak boleh:

- memakai `clean_data.csv` dari baseline untuk semua family lain tanpa penjelasan

### 3.3 Prinsip QLoRA

QLoRA tidak dianggap wajib di awal.

Aturan aman:

- jalankan QLoRA hanya setelah smoke test backend 4-bit lolos
- kalau smoke test gagal, eksperimen utama tetap bisa jalan tanpa QLoRA
- jangan memblok seluruh batch eksperimen hanya karena QLoRA bermasalah

### 3.4 Prinsip evaluasi

Urutan evaluasi yang benar:

1. pilih checkpoint terbaik dari validation
2. cek weak-label / LLM-labelled evaluation
3. cek gold subset / human validation
4. pilih winner final berdasarkan gold subset, bukan weak-label saja

---

## 4. Script yang Dipakai

### 4.1 Setup dan preflight

- [setup_digitalocean_rocm.sh](/c:/Users/alvin/Downloads/skripsi/scripts/setup_digitalocean_rocm.sh)
- [check_qlora_rocm_smoke.py](/c:/Users/alvin/Downloads/skripsi/scripts/check_qlora_rocm_smoke.py)

### 4.2 Runner tuning kecil

- [run_full_finetune_hparam_sweep.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_full_finetune_hparam_sweep.sh)
- [run_peft_hparam_sweep.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_peft_hparam_sweep.sh)
- [export_hparam_sweep_config.py](/c:/Users/alvin/Downloads/skripsi/scripts/export_hparam_sweep_config.py)

### 4.3 Runner training utama

- [run_training_experiments.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_training_experiments.sh)

### 4.4 Runner uncertainty

- [run_uncertainty_experiments.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_uncertainty_experiments.sh)
- [run_uncertainty_retraining.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_uncertainty_retraining.sh)

### 4.5 Evaluasi

- [evaluate.py](/c:/Users/alvin/Downloads/skripsi/src/evaluation/evaluate.py)
- [evaluate_gold_subset.py](/c:/Users/alvin/Downloads/skripsi/src/evaluation/evaluate_gold_subset.py)
- [recommend_epoch_from_epoch_sweep.py](/c:/Users/alvin/Downloads/skripsi/scripts/recommend_epoch_from_epoch_sweep.py)

---

## 5. Hyperparameter dan Aturan Dasar

Landasan lengkapnya ada di [HYPERPARAMETER_LITERATURE_GROUNDED_RULES.md](/c:/Users/alvin/Downloads/skripsi/docs/HYPERPARAMETER_LITERATURE_GROUNDED_RULES.md). Ringkasan yang perlu diingat:

### 5.1 Fixed for fairness

- `max epoch = 15`
- `train batch size = 8`
- format input tetap
- split tetap
- metrik utama checkpoint selection: `validation Macro-F1`

### 5.2 Default utama

Full fine-tuning:

- default LR: `2e-5`
- candidate LR: `2e-5`, `3e-5`, `5e-5`

PEFT:

- default LR: `2e-4`
- candidate LR: `1e-4`, `2e-4`, `3e-4`
- candidate rank: `8`, `16`
- candidate dropout: `0.05`, `0.1`

### 5.3 Metric yang perlu dicatat

Per epoch dan per run, minimal catat:

- validation Macro-F1
- accuracy
- precision
- recall
- train loss
- validation loss
- durasi epoch
- total training time
- best epoch
- best checkpoint

### 5.4 Yang tidak dilakukan

- tidak pakai early stopping
- tidak ubah hyperparameter di tengah satu training run
- tidak tuning semua parameter dengan Optuna

---

## 6. Persiapan Sebelum Login ke Droplet

Sebelum login ke droplet, pastikan hal ini siap di mesin lokal:

- akses SSH ke droplet sudah beres
- repo lokal terbaru sudah benar
- perubahan kode penting sudah di-commit atau minimal tersimpan aman
- Anda tahu branch mana yang akan dipakai
- target eksperimen untuk batch kali ini jelas

Contoh target batch yang aman:

- baseline
- lora
- dora
- adalora

QLoRA:

- baru ikut kalau smoke test lolos

---

## 7. Bootstrap Droplet dari Nol

### 7.1 Login

```bash
ssh root@YOUR_DROPLET_IP
```

Kalau pakai user non-root, sesuaikan.

### 7.2 Install tool dasar

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip tmux unzip
```

### 7.3 Clone repo

```bash
cd ~
git clone https://github.com/alvin-agustio/fintech-review-absa skripsi
cd skripsi
```

Kalau repo private atau pakai remote lain, sesuaikan URL.

### 7.4 Set izin script

```bash
chmod +x scripts/setup_digitalocean_rocm.sh
chmod +x scripts/run_training_experiments.sh
chmod +x scripts/run_uncertainty_experiments.sh
chmod +x scripts/run_uncertainty_retraining.sh
chmod +x scripts/run_full_finetune_hparam_sweep.sh
chmod +x scripts/run_peft_hparam_sweep.sh
```

### 7.5 Jalankan setup ROCm

```bash
bash scripts/setup_digitalocean_rocm.sh
source .venv/bin/activate
```

Script ini dipakai untuk:

- membuat `.venv`
- install PyTorch ROCm
- install dependency proyek
- cek runtime dasar ROCm/PyTorch

---

## 8. Verifikasi Environment

### 8.1 Cek Python dan torch

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
```

Yang diharapkan:

- Python terbaca
- `torch.version.hip` tidak kosong
- `torch.cuda.is_available()` bernilai `True`

### 8.2 Cek ROCm tools

```bash
rocminfo | head
amd-smi list
```

Kalau salah satu tidak ada, catat dulu. Jangan langsung lari ke training besar.

### 8.3 Cek workspace penting

Pastikan folder penting ada:

```bash
ls
ls scripts
ls src/training
ls src/evaluation
ls docs
```

---

## 9. Jalankan QLoRA Smoke Test

Sebelum QLoRA dianggap boleh ikut eksperimen:

```bash
python scripts/check_qlora_rocm_smoke.py
```

Output report akan ditulis ke:

- `data/processed/evaluation/qlora_rocm_smoke_test.json`

### 9.1 Cara membaca hasil smoke test

Kalau hasil `PASS`:

- QLoRA boleh masuk batch run

Kalau hasil `WARN`:

- baca report dulu
- boleh lanjut hanya kalau warning-nya tidak menyentuh backend 4-bit inti

Kalau hasil `FAIL`:

- jangan masukkan QLoRA ke batch utama
- lanjutkan baseline, LoRA, DoRA, dan AdaLoRA dulu

### 9.2 Aturan praktis

Batch utama tidak boleh tertahan hanya karena QLoRA.

---

## 10. Jalankan Tuning Kecil

Bagian ini opsional, tapi sangat disarankan kalau Anda ingin mengunci kandidat LR/r/dropout sebelum run besar.

### 10.1 Full fine-tuning sweep

```bash
bash scripts/run_full_finetune_hparam_sweep.sh
```

Runner ini membaca kandidat dari config dan menulis ringkasan ke:

- `experiments/hparam_sweeps/<family>/sweep_summary.csv`
- `experiments/hparam_sweeps/<family>/sweep_summary.json`

### 10.2 PEFT sweep

```bash
bash scripts/run_peft_hparam_sweep.sh
```

Untuk family tertentu:

```bash
FAMILY=lora bash scripts/run_peft_hparam_sweep.sh
FAMILY=dora bash scripts/run_peft_hparam_sweep.sh
FAMILY=adalora bash scripts/run_peft_hparam_sweep.sh
```

Kalau QLoRA lolos smoke test:

```bash
FAMILY=qlora bash scripts/run_peft_hparam_sweep.sh
```

### 10.3 Aturan memilih pemenang sweep

Gunakan urutan ini:

1. validation Macro-F1 tertinggi
2. kalau mirip, lihat accuracy
3. kalau masih mirip, lihat precision dan recall
4. kalau masih mirip, pilih yang lebih hemat waktu
5. kalau tetap mirip, pilih konfigurasi yang lebih sederhana

### 10.4 Artefak yang perlu dicek

- `sweep_summary.csv`
- `sweep_summary.json`
- `metrics.json` tiap kandidat
- `epoch_log.csv` tiap kandidat

---

## 11. Jalankan Training Utama Without Uncertainty

### 11.1 Jalur utama tanpa QLoRA

```bash
bash scripts/run_training_experiments.sh
```

Ini akan menjalankan family utama:

- baseline
- lora
- dora
- adalora
- retrained
- retrained_lora
- retrained_dora
- retrained_adalora

Catatan:

- istilah `retrained*` di runner ini merujuk pada jalur training family yang sudah disiapkan untuk skenario filtered flow; cek output folder untuk memastikan naming hasil yang keluar

### 11.2 Jalur utama dengan QLoRA

Hanya jika smoke test lolos:

```bash
INCLUDE_QLORA=1 bash scripts/run_training_experiments.sh
```

### 11.3 Artefak minimum yang harus muncul

Per run, minimal pastikan ada:

- `metrics.json`
- `epoch_log.csv`
- `split_manifest.json`
- `split_review_ids.csv`
- `classification_report.txt`
- `test_predictions.csv`

### 11.4 Decision gate setelah training utama

Sebelum lanjut ke uncertainty, cek:

- apakah semua family utama selesai tanpa crash
- apakah `metrics.json` ada
- apakah `epoch_log.csv` ada
- apakah `best_epoch` dan `best_checkpoint` tercatat
- apakah split manifest ada

Kalau ada run yang gagal, perbaiki dulu. Jangan lanjut buta ke uncertainty.

---

## 12. Jalankan Uncertainty Family-Aware

### 12.1 Jalur utama

```bash
bash scripts/run_uncertainty_experiments.sh
```

### 12.2 Kalau QLoRA ikut

```bash
INCLUDE_QLORA=1 bash scripts/run_uncertainty_experiments.sh
```

### 12.3 Output yang diharapkan

Contoh:

- `data/processed/uncertainty/baseline/epoch_15/mc_predictions.csv`
- `data/processed/uncertainty/baseline/epoch_15/mc_summary.json`
- `data/processed/noise/baseline/epoch_15/clean_data.csv`
- `data/processed/noise/baseline/epoch_15/noisy_data.csv`
- `data/processed/noise/baseline/epoch_15/noise_summary.json`

Dan pola yang sama untuk:

- `lora`
- `dora`
- `adalora`
- `qlora` jika dijalankan

### 12.4 Decision gate uncertainty

Sebelum retraining uncertainty, cek:

- apakah setiap family punya `mc_summary.json`
- apakah setiap family punya `clean_data.csv`
- apakah threshold dan metric uncertainty tercatat
- apakah tidak ada family yang diam-diam memakai clean subset family lain

---

## 13. Jalankan Retraining pada Clean Subset

### 13.1 Jalur utama

```bash
bash scripts/run_uncertainty_retraining.sh
```

### 13.2 Kalau QLoRA ikut

```bash
INCLUDE_QLORA=1 bash scripts/run_uncertainty_retraining.sh
```

### 13.3 Makna tahap ini

Tahap ini menjalankan model ulang pada clean subset family masing-masing.

Contoh konsep yang benar:

- baseline memakai clean subset baseline
- lora memakai clean subset lora
- dora memakai clean subset dora
- adalora memakai clean subset adalora

### 13.4 Yang tidak boleh

- baseline clean subset dipakai ke semua family tanpa alasan metodologis
- subset family tercampur karena path salah

---

## 14. Jalankan Evaluasi Akhir

### 14.1 Weak-label / LLM-labelled evaluation

```bash
python src/evaluation/evaluate.py
```

Output penting:

- `data/processed/evaluation/evaluation_summary.json`
- `data/processed/evaluation/evaluation_detailed.json`
- `data/processed/evaluation/model_comparison_table.csv`
- `data/processed/evaluation/comparison_group_best.csv`

### 14.2 Human / gold subset evaluation

```bash
python src/evaluation/evaluate_gold_subset.py
```

Output penting:

- `data/processed/diamond/evaluation/gold_evaluation_overview.csv`
- `data/processed/diamond/evaluation/gold_evaluation_group_best.csv`

### 14.3 Cara membaca hasil

Jangan langsung pilih winner dari weak-label saja.

Urutan baca yang benar:

1. lihat comparison table weak-label
2. lihat comparison group best
3. cek hasil gold subset
4. pilih winner final berdasarkan gold subset

---

## 15. Matriks Eksperimen yang Diinginkan

Target bentuk eksperimen akhirnya:

- `baseline`
- `baseline_unc`
- `lora`
- `lora_unc`
- `dora`
- `dora_unc`
- `adalora`
- `adalora_unc`
- `qlora`
- `qlora_unc`

Kalau QLoRA tidak lolos smoke test, matriks tetap sah tanpa dua baris terakhir.

---

## 16. Artefak yang Wajib Ada Sebelum Dianggap Selesai

Per run training:

- `metrics.json`
- `epoch_log.csv`
- `split_manifest.json`
- `split_review_ids.csv`

Per run uncertainty:

- `mc_predictions.csv`
- `mc_summary.json`
- `clean_data.csv`
- `noisy_data.csv`
- `noise_summary.json`

Per batch evaluasi:

- weak-label evaluation output
- gold subset evaluation output

Kalau salah satu hilang, anggap batch belum rapi.

---

## 17. Decision Tree Sederhana Saat Ada Masalah

### 17.1 Torch atau ROCm tidak kebaca

Lakukan:

1. cek hasil setup script
2. cek `python -c "import torch; ..."`
3. cek `rocminfo`
4. jangan lanjut training besar sampai ini beres

### 17.2 QLoRA smoke test gagal

Lakukan:

1. keluarkan QLoRA dari batch
2. lanjut baseline, LoRA, DoRA, AdaLoRA
3. kembali ke QLoRA nanti sebagai eksperimen terpisah

### 17.3 Training salah satu family gagal

Lakukan:

1. cek path output run itu
2. cek `metrics.json` ada atau tidak
3. cek error log terminal
4. rerun family itu saja jika perlu
5. jangan lanjut ke uncertainty kalau output training family belum lengkap

### 17.4 Uncertainty output tidak lengkap

Lakukan:

1. cek `mc_summary.json`
2. cek `noise_summary.json`
3. cek apakah clean subset tertulis ke family yang benar
4. rerun uncertainty family itu

### 17.5 Evaluasi tidak menangkap semua model

Lakukan:

1. cek struktur folder model
2. cek `metrics.json`
3. cek naming family dan regime
4. rerun evaluator

---

## 18. Arsipkan Hasil Sebelum Droplet Dihancurkan

### 18.1 Inti evaluasi

```bash
mkdir -p exports
tar -czf exports/skripsi_eval_core.tar.gz \
  data/processed/evaluation \
  data/processed/uncertainty \
  data/processed/noise
```

### 18.2 Laporan eksperimen ringkas

```bash
tar -czf exports/skripsi_experiment_reports.tar.gz \
  models/baseline/epoch_15/metrics.json \
  models/lora/epoch_15/metrics.json \
  models/dora/epoch_15/metrics.json \
  models/adalora/epoch_15/metrics.json \
  models/retrained/epoch_15/metrics.json \
  models/retrained_lora/epoch_15/metrics.json \
  models/retrained_dora/epoch_15/metrics.json \
  models/retrained_adalora/epoch_15/metrics.json
```

Kalau QLoRA ikut dan berhasil:

```bash
tar -rzf exports/skripsi_experiment_reports.tar.gz \
  models/qlora/epoch_15/metrics.json \
  models/retrained_qlora/epoch_15/metrics.json
```

Kalau `tar -rzf` tidak cocok di image tertentu, buat arsip terpisah.

### 18.3 Simpan model final terbaik

Sesuaikan dengan winner final Anda. Contoh:

```bash
tar -czf exports/skripsi_best_model.tar.gz \
  models/retrained_lora/epoch_15/model
```

### 18.4 Simpan metadata reproducibility

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

---

## 19. Download Hasil ke Lokal

Contoh dari Windows PowerShell:

```powershell
scp -i $HOME\.ssh\id_ed25519_nopass root@YOUR_DROPLET_IP:/root/skripsi/exports/skripsi_eval_core.tar.gz C:\Users\alvin\Downloads\
scp -i $HOME\.ssh\id_ed25519_nopass root@YOUR_DROPLET_IP:/root/skripsi/exports/skripsi_experiment_reports.tar.gz C:\Users\alvin\Downloads\
scp -i $HOME\.ssh\id_ed25519_nopass root@YOUR_DROPLET_IP:/root/skripsi/exports/skripsi_best_model.tar.gz C:\Users\alvin\Downloads\
scp -i $HOME\.ssh\id_ed25519_nopass root@YOUR_DROPLET_IP:/root/skripsi/exports/skripsi_metadata_bundle.tar.gz C:\Users\alvin\Downloads\
```

---

## 20. Verifikasi Lokal

Jangan destroy droplet sebelum arsip lokal dicek.

Contoh:

```powershell
tar -tf C:\Users\alvin\Downloads\skripsi_eval_core.tar.gz
tar -tf C:\Users\alvin\Downloads\skripsi_experiment_reports.tar.gz
tar -tf C:\Users\alvin\Downloads\skripsi_best_model.tar.gz
tar -tf C:\Users\alvin\Downloads\skripsi_metadata_bundle.tar.gz
```

Kalau perlu, ekstrak satu arsip dulu untuk memastikan isinya benar.

---

## 21. Kapan Droplet Aman Dihancurkan

Droplet aman di-destroy hanya kalau:

- semua artefak penting sudah dibuat
- semua arsip penting sudah berhasil dipindah ke lokal
- arsip lokal bisa dibuka
- model final terbaik sudah ikut tersimpan
- metadata reproducibility sudah tersimpan
- tidak ada eksperimen penting yang masih hanya hidup di droplet

Kalau salah satu belum selesai, jangan destroy dulu.

---

## 22. Rekomendasi Jalur Eksekusi yang Paling Aman

Kalau mau paling aman dan tidak terlalu berisiko:

### Tahap 1

- setup ROCm
- verifikasi GPU
- smoke test QLoRA

### Tahap 2

- tuning kecil baseline
- tuning kecil LoRA
- tuning kecil DoRA
- tuning kecil AdaLoRA

### Tahap 3

- training utama baseline
- training utama LoRA
- training utama DoRA
- training utama AdaLoRA

### Tahap 4

- uncertainty baseline
- uncertainty LoRA
- uncertainty DoRA
- uncertainty AdaLoRA

### Tahap 5

- retraining uncertainty baseline
- retraining uncertainty LoRA
- retraining uncertainty DoRA
- retraining uncertainty AdaLoRA

### Tahap 6

- evaluate weak-label
- evaluate gold subset
- pilih winner sementara

### Tahap 7

- kalau QLoRA lolos smoke test, baru masukkan QLoRA ke flow terpisah

Ini membuat eksperimen utama tetap jalan walaupun QLoRA ternyata belum stabil.

---

## 23. TL;DR yang Sangat Singkat

Kalau nanti Anda cuma butuh ingat garis besarnya:

1. setup ROCm
2. aktifkan `.venv`
3. cek torch dan ROCm
4. jalankan smoke test QLoRA
5. jalankan tuning kecil jika perlu
6. jalankan training utama
7. jalankan uncertainty
8. jalankan retraining uncertainty
9. jalankan evaluasi weak-label dan gold subset
10. arsipkan hasil
11. download hasil
12. verifikasi lokal
13. baru destroy droplet

---

## 24. Referensi Internal yang Perlu Diingat

- [DIGITALOCEAN_READY_MANIFEST.md](/c:/Users/alvin/Downloads/skripsi/docs/DIGITALOCEAN_READY_MANIFEST.md)
- [DIGITALOCEAN_TRAINING_RUNBOOK.md](/c:/Users/alvin/Downloads/skripsi/docs/DIGITALOCEAN_TRAINING_RUNBOOK.md)
- [UNCERTAINTY_DIAMOND_STANDARD_RULES.md](/c:/Users/alvin/Downloads/skripsi/docs/UNCERTAINTY_DIAMOND_STANDARD_RULES.md)
- [HYPERPARAMETER_LITERATURE_GROUNDED_RULES.md](/c:/Users/alvin/Downloads/skripsi/docs/HYPERPARAMETER_LITERATURE_GROUNDED_RULES.md)

Kalau ada kebingungan saat run, kembali ke dokumen ini dulu.
