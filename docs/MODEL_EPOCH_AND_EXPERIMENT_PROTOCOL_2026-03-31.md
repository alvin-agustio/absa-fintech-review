# Model Epoch and Experiment Protocol

Freeze date: 2026-03-31

Status: canonical protocol for model selection after taxonomy freeze

Version: `experiment-protocol-v1.0`

## Purpose

Dokumen ini mengunci urutan eksperimen model setelah taxonomy dashboard dianggap final. Tujuannya adalah membuat pemilihan epoch, evaluasi candidate models, dan keputusan model final menjadi explainable untuk skripsi.

Dokumen ini dipakai bersama artefak repo yang sudah ada, terutama:

- `src/training/train_baseline.py`
- `src/training/train_lora.py`
- `src/training/retrain_filtered.py`
- `src/training/train_lora_filtered.py`
- `src/evaluation/evaluate.py`
- `scripts/run_training_experiments.ps1`
- `scripts/recommend_epoch_from_epoch_sweep.py`

## Canonical Decision Order

Urutan keputusan yang dikunci adalah:

1. Freeze taxonomy dan schema aspect/issue.
2. Calibrate epoch using validation `F1 Macro`.
3. Compare candidate families on the same data split.
4. Evaluate the winning candidates on the gold subset.
5. Choose the final model based on human-validated evidence, not weak-label score alone.

## What Is Fixed

Yang dianggap stabil selama protocol ini aktif:

- backbone utama tetap `indobenchmark/indobert-base-p1`
- label space tetap `Negative`, `Neutral`, `Positive`
- aspect-conditioned input tetap memakai format `"[ASPECT=...] review_text"`
- validation metric utama tetap `F1 Macro`
- sweep epoch referensi tetap `3`, `5`, dan `8`

## Epoch Selection Logic

Epoch tidak dipilih secara intuitif. Epoch dipilih dari hasil validasi dengan aturan berikut:

1. Jalankan sweep epoch yang sama untuk semua kandidat yang sedang dibandingkan.
2. Simpan checkpoint terbaik pada setiap run dengan `load_best_model_at_end=True`.
3. Gunakan `F1 Macro` validasi sebagai metrik utama.
4. Jika dua run sangat dekat, pakai `Accuracy` sebagai tie-breaker kedua.
5. Jika masih tie, pakai `Training Time` yang lebih kecil.
6. Jika semua metrik masih sama, pilih epoch yang lebih kecil agar lebih hemat compute dan lebih mudah dijelaskan.

Interpretasi praktis:

- `epoch 3` dipakai sebagai awal pembacaan kurva.
- `epoch 5` dipakai sebagai titik tengah.
- `epoch 8` dipakai sebagai titik eksplorasi yang lebih panjang.
- epoch terbaik adalah epoch yang paling kuat di validation, bukan yang paling besar.

## Early Stopping and Validation Logic

Repo saat ini sudah memakai validation per epoch dan mengembalikan checkpoint terbaik pada akhir run. Itu berarti:

- setiap run sudah punya mekanisme pemilihan checkpoint terbaik via validation
- test set tidak boleh dipakai untuk memilih epoch
- test set hanya dipakai untuk laporan akhir setelah checkpoint dipilih

Jika ingin menambahkan early stopping eksplisit pada eksperimen berikutnya, aturan yang direkomendasikan adalah:

- monitor `F1 Macro`
- gunakan patience pendek, misalnya `2` epoch
- restore best checkpoint at end
- jangan ubah validation split setelah sweep dimulai

Kalau early stopping tidak diaktifkan, protocol ini tetap valid karena sweep epoch 3/5/8 berfungsi sebagai coarse search space yang jelas.

## Candidate Model Ladder

Candidate models dievaluasi dalam dua kelompok:

### Reference family

Ini dipakai untuk mengunci baseline metodologis:

- `baseline`
- `lora`

### Extended family

Ini dipakai setelah landasan epoch sudah jelas:

- `DoRA`
- `QLoRA`
- `AdaLoRA`
- `retrained`
- `retrained_lora`

Jika resource terbatas, urutan yang paling aman adalah:

1. baseline
2. LoRA
3. DoRA
4. QLoRA
5. AdaLoRA
6. retrained variants

## Evaluation Ladder

Evaluasi dilakukan bertahap, bukan langsung loncat ke model final.

### Step 1. Validation calibration

Tujuan:

- memilih epoch yang paling stabil
- menghindari keputusan epoch yang terasa arbitrer

Output:

- per-family best epoch
- overall best epoch candidate

### Step 2. Weak-label comparison

Tujuan:

- membandingkan kandidat pada benchmark otomatis yang besar
- melihat apakah PEFT mengalahkan full fine-tuning

Output:

- `accuracy`
- `F1 Macro`
- `F1 Weighted`
- training time

### Step 3. Gold subset comparison

Tujuan:

- memeriksa apakah model yang bagus di weak-label juga bagus di human-validated data
- menghindari model yang menang di label lemah tetapi tidak stabil di data manual

Output:

- `sentiment_accuracy_present`
- `sentiment_f1_macro_present`
- error patterns on rows without aspect presence

### Step 4. Final model selection

Final model dipilih dari gold subset, dengan urutan prioritas:

1. `F1 Macro` gold subset
2. `Accuracy` gold subset
3. training efficiency
4. deployment practicality

## Execution Order

Urutan eksekusi yang disarankan:

1. Pastikan taxonomy freeze sudah aktif.
2. Jalankan sweep kandidat reference family.
3. Jalankan helper rekomendasi epoch.
4. Kunci epoch terbaik untuk eksperimen utama.
5. Jalankan sweep candidate families pada epoch yang sudah jelas.
6. Jalankan `evaluate.py` untuk weak-label comparison.
7. Jalankan evaluasi gold subset.
8. Pilih final model untuk dashboard dan penulisan skripsi.

## Ready-To-Use Commands

```powershell
.\.venv\Scripts\python.exe src\evaluation\evaluate.py
.\.venv\Scripts\python.exe scripts\recommend_epoch_from_epoch_sweep.py
.\scripts\run_training_experiments.ps1
```

## Reporting Rule For Thesis

Saat menulis metodologi, jelaskan dengan kalimat sederhana:

- epoch dipilih dari validation `F1 Macro`
- checkpoint terbaik per run diambil dari validation
- weak-label test dipakai untuk benchmark otomatis
- gold subset dipakai untuk validasi manusia
- final model dipilih dari gold subset, bukan dari weak-label saja

## Practical Checklist

Gunakan checklist ini sebelum mengunci hasil eksperimen:

- taxonomy sudah freeze
- train/validation/test split tidak berubah
- metric utama sudah ditetapkan: `F1 Macro`
- checkpoint terbaik per run sudah tersimpan
- hasil weak-label sudah dibandingkan
- hasil gold subset sudah dibandingkan
- final model sudah dipilih dari gold subset
- artifact evaluasi sudah tersimpan untuk paper dan dashboard

## Limitations

Protocol ini sengaja sederhana supaya explainable. Artinya:

- tidak memaksakan multi-seed sweep besar
- tidak menuntut benchmark model di luar keluarga yang relevan untuk repo ini
- tidak memakai test set untuk memilih epoch
- tidak mengubah taxonomy lagi di tengah eksperimen

Kalau nanti ada eksperimen baru yang mengubah asumsi di atas, maka itu harus diperlakukan sebagai versi protocol baru.
