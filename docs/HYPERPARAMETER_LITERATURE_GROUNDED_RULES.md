# Hyperparameter Literature-Grounded Rules

Status: active reference for the next training cycle

Dokumen ini dipakai untuk menjelaskan kenapa angka hyperparameter tidak dipilih secara asal. Tujuannya bukan menyalin satu angka mentah dari paper, tetapi mengambil rentang yang wajar dari literatur lalu menguncinya secara konsisten untuk repo ini.

## Prinsip Dasar

Hyperparameter dibagi menjadi tiga kelompok:

- `fixed for fairness`: dipakai sama untuk semua model agar perbandingan adil
- `literature-informed default`: dipilih dari praktik umum atau paper primer
- `small tuning space`: tidak dituning liar; hanya dicari pada rentang kecil yang masih masuk akal

## Fixed for Fairness

Aturan yang dikunci untuk eksperimen utama:

- `max epoch = 15`
- `train batch size = 8`
- format input tetap `"[ASPECT=...] review_text"`
- split `train/validation/test` tetap
- metrik utama untuk pemilihan checkpoint adalah `validation Macro-F1`

Catatan:

- Literatur BERT dan RoBERTa sering memakai batch size `16` atau `32`, tetapi repo ini memakai `8` sebagai penyesuaian resource agar semua family model masih bisa dibandingkan secara adil di GPU yang sama.
- Jadi angka `8` bukan angka dari paper, melainkan angka operasional yang dipilih untuk fairness dan keterbatasan memori.

## Literature-Informed Default

### Full Fine-Tuning

Default yang dipakai:

- `learning rate = 2e-5`

Ruang tuning kecil:

- `2e-5`
- `3e-5`
- `5e-5`

Landasan:

- BERT merekomendasikan fine-tuning dengan learning rate `2e-5`, `3e-5`, atau `5e-5`, dan epoch `2`, `3`, atau `4`.
- RoBERTa juga memakai sweep kecil learning rate sekitar `1e-5` sampai `3e-5` pada fine-tuning.

### LoRA / DoRA / AdaLoRA

Default yang dipakai:

- `learning rate = 2e-4`
- `rank = 16`
- `alpha = 32`
- `dropout = 0.1`

Ruang tuning kecil:

- learning rate: `1e-4`, `2e-4`, `3e-4`
- rank: `8`, `16`
- dropout: `0.05`, `0.1`

Landasan:

- LoRA dan turunannya umumnya memakai learning rate yang lebih besar daripada full fine-tuning.
- DoRA pada contoh eksperimen juga sering memakai kombinasi sekitar `r=16`, `alpha=32`, dan learning rate `1e-4` sampai `2e-4`.
- AdaLoRA tetap memakai dasar LoRA, tetapi menambah pengaturan budget rank yang adaptif.

### QLoRA

Default yang dipakai:

- `learning rate = 2e-5`
- `rank = 16`
- `alpha = 32`
- `dropout = 0.1`
- quantization: `NF4`
- `double quantization = True`
- compute dtype: `float16`

Catatan penting:

- Untuk QLoRA, yang paling kuat dari literatur adalah resep quantization, bukan menyalin semua angka mentah dari paper LLM besar.
- Karena repo ini memakai IndoBERT untuk klasifikasi, angka learning rate QLoRA dibuat lebih konservatif dan diperlakukan sebagai jalur opsional yang perlu smoke test di droplet NVIDIA/CUDA.

## AdaLoRA-Specific Rules

Default yang dipakai:

- `init_r = 12`
- `target_r = 8`
- `tinit = 0`
- `tfinal = 0`
- `deltaT = 1`
- `beta1 = 0.85`
- `beta2 = 0.85`
- `orth_reg_weight = 0.5`

Landasan:

- Ini mengikuti konfigurasi PEFT/AdaLoRA yang umum dipakai dan menjaga agar budget rank benar-benar berubah selama training.
- `total_step` harus dihitung dari ukuran dataset, batch size, dan jumlah epoch, bukan dipilih manual.

## Apa yang Tidak Dilakukan

Dokumen ini sengaja tidak mewajibkan:

- Optuna untuk semua hyperparameter
- tuning besar untuk semua model
- pengubahan hyperparameter di tengah training run

Alasannya:

- eksperimen akan terlalu besar
- interpretasi akan jadi sulit
- fairness antar family model justru bisa memburuk

## Implementasi di Code

Angka-angka ini sekarang dikunci di `config.py`:

- `TRAIN_MAX_EPOCHS`
- `TRAIN_BATCH_SIZE`
- `FULL_FINETUNE_DEFAULT_LR`
- `FULL_FINETUNE_LR_CANDIDATES`
- `PEFT_DEFAULT_LR`
- `PEFT_LR_CANDIDATES`
- `PEFT_R_CANDIDATES`
- `PEFT_DROPOUT_CANDIDATES`

Lalu dipakai langsung oleh:

- `src/training/train_baseline.py`
- `src/training/retrain_filtered.py`
- `src/training/train_lora.py`
- `src/training/train_lora_filtered.py`
- `src/training/peft_family_utils.py`

## Runner Sweep

Untuk menjalankan sweep kecil secara otomatis, gunakan:

```powershell
.\scripts\run_full_finetune_hparam_sweep.ps1 -Family baseline
.\scripts\run_peft_hparam_sweep.ps1 -Family lora
.\scripts\run_peft_hparam_sweep.ps1 -Family dora
.\scripts\run_peft_hparam_sweep.ps1 -Family adalora
```

Untuk versi uncertainty retraining:

```powershell
.\scripts\run_full_finetune_hparam_sweep.ps1 -Family retrained -CleanCsv data/processed/noise/baseline/epoch_15/clean_data.csv
.\scripts\run_peft_hparam_sweep.ps1 -Family lora -Filtered -CleanCsv data/processed/noise/lora/epoch_15/clean_data.csv
```

Untuk `QLoRA`, jalankan hanya di droplet NVIDIA/CUDA yang sudah lolos smoke test:

```powershell
python scripts/check_qlora_rocm_smoke.py
.\scripts\run_peft_hparam_sweep.ps1 -Family qlora
```

Semua runner menulis ringkasan ke:

- `experiments/hparam_sweeps/.../sweep_summary.csv`
- `experiments/hparam_sweeps/.../sweep_summary.json`

## Sumber Primer

- BERT: https://arxiv.org/abs/1810.04805
- RoBERTa: https://arxiv.org/abs/1907.11692
- LoRA: https://arxiv.org/abs/2106.09685
- AdaLoRA: https://arxiv.org/abs/2303.10512
- QLoRA: https://arxiv.org/abs/2305.14314
- DoRA: https://arxiv.org/abs/2402.09353

## Kalimat Singkat untuk Metodologi

> Hyperparameter tidak dipilih secara arbitrer, melainkan berdasarkan kombinasi antara praktik umum pada literatur primer, kebutuhan fairness antar model, keterbatasan komputasi, dan ruang tuning kecil yang tetap dikontrol pada validation set.
