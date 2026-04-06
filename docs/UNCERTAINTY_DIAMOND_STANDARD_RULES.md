# Uncertainty Diamond Standard Rules

Status: active standard for uncertainty-aware experiments

Dokumen ini mengunci aturan untuk eksperimen uncertainty-aware agar perbandingan antara:

- Baseline
- Berbagai PEFT
- Baseline dengan Uncertainty
- Berbagai PEFT dengan Uncertainty

tetap adil, mudah dijelaskan, dan tidak mudah dipatahkan secara metodologis.

## Core Principle

Uncertainty tidak boleh diperlakukan sebagai trik tambahan yang hanya dipakai untuk satu model. Uncertainty harus diperlakukan sebagai faktor eksperimen yang eksplisit.

Artinya:

- `without uncertainty` dan `with uncertainty` adalah dua kondisi yang berbeda
- setiap family model harus punya jalur uncertainty sendiri
- hasil uncertainty satu family tidak boleh dipaksakan menjadi clean subset global untuk semua family lain

## Diamond Standard Naming

Gunakan struktur nama yang konsisten:

- weak-label model:
  - `baseline`
  - `lora`
  - `dora`
  - `qlora`
  - `adalora`
- uncertainty artifacts:
  - `data/processed/uncertainty/<family>/<run_name>/mc_predictions.csv`
  - `data/processed/uncertainty/<family>/<run_name>/mc_summary.json`
- noise artifacts:
  - `data/processed/noise/<family>/<run_name>/clean_data.csv`
  - `data/processed/noise/<family>/<run_name>/noisy_data.csv`
  - `data/processed/noise/<family>/<run_name>/noise_summary.json`

Contoh:

- `data/processed/uncertainty/lora/epoch_15/mc_predictions.csv`
- `data/processed/noise/lora/epoch_15/clean_data.csv`

## Family-Aware Rule

Setiap family model harus menjalankan uncertainty pada modelnya sendiri.

Contoh yang benar:

- `baseline -> MC Dropout -> clean subset baseline -> baseline_unc`
- `lora -> MC Dropout -> clean subset lora -> lora_unc`
- `dora -> MC Dropout -> clean subset dora -> dora_unc`

Contoh yang tidak boleh:

- memakai `clean_data.csv` hasil baseline uncertainty untuk semua family PEFT tanpa penjelasan

## Two Validation Rule

Setiap kandidat model harus melewati dua validasi:

1. `LLM-labelled / weak-label validation`
2. `Human subset / gold validation`

Urutan keputusan:

1. pilih checkpoint terbaik dari validation
2. bandingkan weak-label benchmark
3. bandingkan human subset
4. pilih model final dari human subset, bukan weak-label saja

## Split Discipline

Semua training dan retraining harus memakai split yang sama secara konsep:

- train
- validation
- test

Aturan:

- split dilakukan di level `review_id`, bukan baris `(review, aspect)` jika itu berisiko leakage
- test set tidak boleh dipakai untuk memilih epoch
- gold subset tidak boleh dipakai untuk tuning threshold utama

## Uncertainty Metric Rule

Uncertainty metric yang dipakai harus dicatat eksplisit.

Minimal catat:

- `uncertainty_entropy`
- `uncertainty_variance`
- `num_mc`
- `high_uncertainty_quantile`
- threshold numerik yang dihasilkan

Jika filtering memakai `uncertainty_entropy`, maka itu harus konsisten tertulis di summary dan paper.

## Threshold Guardrail

Threshold uncertainty tidak boleh diasumsikan otomatis optimal untuk semua family.

Aturan aman:

- threshold default boleh sama sebagai titik awal
- tetapi hasil tiap family harus tetap dicatat terpisah
- jika threshold berbeda per family, itu harus dinyatakan eksplisit

## Stochasticity Guardrail for PEFT

Untuk PEFT, jangan langsung menganggap MC Dropout valid tanpa pengecekan.

Minimal guardrail:

- pastikan model memang punya dropout modules aktif
- catat jumlah dropout modules
- jika tidak ada dropout aktif, jangan klaim MC Dropout valid

## Fair Efficiency Rule

Kalau ingin membandingkan efisiensi:

- batch size harus sama atau perbedaannya dilaporkan jelas
- perangkat komputasi harus sama
- max epoch harus sama
- metric selection harus sama
- logging waktu per epoch harus ada

Tanpa itu, klaim efisiensi PEFT bisa dipertanyakan.

## Artifact Completeness Rule

Setiap run uncertainty-aware minimal harus menghasilkan:

- training `metrics.json`
- `epoch_log.csv`
- `split_manifest.json`
- `mc_predictions.csv`
- `mc_summary.json`
- `clean_data.csv`
- `noise_summary.json`
- weak-label evaluation output
- gold subset evaluation output

## Experiment Matrix Rule

Struktur eksperimen yang ideal:

- `baseline`
- `baseline_unc`
- `lora`
- `lora_unc`
- `dora`
- `dora_unc`
- `qlora`
- `qlora_unc`
- `adalora`
- `adalora_unc`

Dengan begitu, pembaca bisa langsung melihat dua faktor:

- family model
- uncertainty on/off

## Devil's Advocate Checklist

Sebelum hasil dianggap final, cek pertanyaan ini:

- Apakah clean subset untuk tiap family berasal dari model family yang sama?
- Apakah split train/val/test bebas leakage review?
- Apakah epoch dipilih dari validation, bukan test?
- Apakah uncertainty threshold ditetapkan dengan aturan yang jelas?
- Apakah PEFT uncertainty benar-benar stochastic?
- Apakah batch size/device sama saat membandingkan efisiensi?
- Apakah winner weak-label sama atau berbeda dari winner human subset?

## Simple TL;DR

Aturan emasnya sederhana:

- uncertainty harus family-aware
- validation dan human subset harus tetap dipisah
- artifact harus lengkap
- naming harus konsisten
- fairness harus dijaga sejak split sampai evaluasi akhir
