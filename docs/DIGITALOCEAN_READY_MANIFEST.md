# DigitalOcean-Ready Manifest

Dokumen ini adalah manifest final untuk menjalankan repo ini secara native di DigitalOcean Linux, khususnya untuk droplet AMD MI300X / ROCm.

## Script Wajib

### Setup dan preflight

- [setup_digitalocean_rocm.sh](/c:/Users/alvin/Downloads/skripsi/scripts/setup_digitalocean_rocm.sh)
- [check_qlora_rocm_smoke.py](/c:/Users/alvin/Downloads/skripsi/scripts/check_qlora_rocm_smoke.py)

### Runner utama

- [run_training_experiments.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_training_experiments.sh)
- [run_uncertainty_experiments.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_uncertainty_experiments.sh)
- [run_uncertainty_retraining.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_uncertainty_retraining.sh)

### Runner tuning kecil

- [run_full_finetune_hparam_sweep.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_full_finetune_hparam_sweep.sh)
- [run_peft_hparam_sweep.sh](/c:/Users/alvin/Downloads/skripsi/scripts/run_peft_hparam_sweep.sh)
- [export_hparam_sweep_config.py](/c:/Users/alvin/Downloads/skripsi/scripts/export_hparam_sweep_config.py)

### Evaluasi

- [evaluate.py](/c:/Users/alvin/Downloads/skripsi/src/evaluation/evaluate.py)
- [evaluate_gold_subset.py](/c:/Users/alvin/Downloads/skripsi/src/evaluation/evaluate_gold_subset.py)
- [recommend_epoch_from_epoch_sweep.py](/c:/Users/alvin/Downloads/skripsi/scripts/recommend_epoch_from_epoch_sweep.py)

## Worker Script yang Harus Ada

### Training core

- [train_baseline.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_baseline.py)
- [train_lora.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_lora.py)
- [train_dora.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_dora.py)
- [train_adalora.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_adalora.py)
- [train_qlora.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_qlora.py)
- [retrain_filtered.py](/c:/Users/alvin/Downloads/skripsi/src/training/retrain_filtered.py)
- [train_lora_filtered.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_lora_filtered.py)
- [train_dora_filtered.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_dora_filtered.py)
- [train_adalora_filtered.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_adalora_filtered.py)
- [train_qlora_filtered.py](/c:/Users/alvin/Downloads/skripsi/src/training/train_qlora_filtered.py)
- [peft_family_utils.py](/c:/Users/alvin/Downloads/skripsi/src/training/peft_family_utils.py)
- [run_utils.py](/c:/Users/alvin/Downloads/skripsi/src/training/run_utils.py)

### Uncertainty core

- [predict_mc_dropout.py](/c:/Users/alvin/Downloads/skripsi/src/evaluation/predict_mc_dropout.py)
- [detect_label_noise.py](/c:/Users/alvin/Downloads/skripsi/src/evaluation/detect_label_noise.py)

## Urutan Pakai di Droplet

1. Jalankan setup ROCm:

```bash
bash scripts/setup_digitalocean_rocm.sh
source .venv/bin/activate
```

2. Jalankan smoke test `QLoRA`:

```bash
python scripts/check_qlora_rocm_smoke.py
```

3. Jalankan tuning kecil jika diperlukan:

```bash
bash scripts/run_full_finetune_hparam_sweep.sh
bash scripts/run_peft_hparam_sweep.sh
```

4. Jalankan training utama:

```bash
bash scripts/run_training_experiments.sh
```

Jika ingin ikut `QLoRA`:

```bash
INCLUDE_QLORA=1 bash scripts/run_training_experiments.sh
```

5. Jalankan uncertainty:

```bash
bash scripts/run_uncertainty_experiments.sh
```

Jika ingin ikut `QLoRA`:

```bash
INCLUDE_QLORA=1 bash scripts/run_uncertainty_experiments.sh
```

6. Jalankan retraining uncertainty:

```bash
bash scripts/run_uncertainty_retraining.sh
```

Jika ingin ikut `QLoRA`:

```bash
INCLUDE_QLORA=1 bash scripts/run_uncertainty_retraining.sh
```

7. Jalankan evaluasi akhir:

```bash
python src/evaluation/evaluate.py
python src/evaluation/evaluate_gold_subset.py
```

## Catatan Operasional

- Untuk MI300X, gunakan runner `.sh`, bukan `.ps1`.
- `QLoRA` tetap dianggap opsional sampai smoke test 4-bit lolos.
- Batch size training dikunci `8` untuk fairness antar model.
- Epoch maksimum dikunci `15`.
- Semua keputusan model tetap dibaca dari validation dulu, lalu dikonfirmasi di gold subset.
