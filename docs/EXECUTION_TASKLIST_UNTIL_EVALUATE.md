# Execution Tasklist Until Evaluate

Dokumen ini membatasi scope sampai tahap evaluasi model. Dashboard, inference production, dan diamond-subset curation belum masuk.

## Tujuan

Menjalankan pipeline eksperimen secara rapi dari dataset final sampai perbandingan model:

1. Baseline full fine-tune
2. LoRA fine-tune
3. MC Dropout uncertainty
4. Noise detection
5. Retrain on filtered data
6. Final evaluation

## Preflight Checklist

- [ ] Pastikan virtual environment aktif.
- [ ] Pastikan dataset utama tersedia di `data/processed/dataset_absa.csv`.
- [ ] Pastikan disk masih cukup untuk menyimpan checkpoint model.
- [ ] Pastikan koneksi internet tersedia untuk download model Hugging Face jika cache belum ada.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Status Kompatibilitas Pipeline

Kompatibilitas utama untuk tahap uncertainty sampai evaluate sudah dirapikan:

- [x] `train_baseline.py` sudah siap untuk 3 aspek dan default ke `data/processed/dataset_absa.csv`.
- [x] `train_lora.py` sudah siap untuk 3 aspek.
- [x] `predict_mc_dropout.py` sudah mendukung `risk`, `trust`, dan `service`.
- [x] Output uncertainty sekarang default ke `data/processed/uncertainty`.
- [x] Output noise detection sekarang default ke `data/processed/noise`.
- [x] `train_baseline.py` dan `retrain_filtered.py` sekarang juga menghasilkan `metrics.json`, `classification_report.txt`, dan `test_predictions.csv` selain nama file legacy.
- [x] `retrain_filtered.py` sekarang default ke `models/retrained`.
- [x] `evaluate.py` sekarang bisa membaca artefak baru maupun nama file legacy.

Catatan: `evaluate.py` tetap akan melewati eksperimen yang belum dijalankan. Jadi output `[SKIP]` masih normal kalau folder model belum berisi metrik.

## Urutan Eksekusi

### 1. Freeze dan validasi dataset

Tujuan: memastikan satu sumber data dipakai konsisten oleh semua eksperimen.

- [ ] Gunakan hanya `data/processed/dataset_absa.csv` sebagai dataset training.
- [ ] Catat total review, total labeled, distribusi aspek, dan distribusi label.
- [ ] Jangan ubah isi dataset di tengah eksperimen baseline, LoRA, dan retrain.

Kriteria selesai:

- [ ] Dataset final sudah tetap dan siap dipakai seluruh eksperimen.

### 2. Train LoRA terlebih dahulu

Tujuan: mendapatkan model awal yang lebih efisien dan cepat sebagai kandidat utama.

Perintah:

```powershell
.\.venv\Scripts\python.exe train_lora.py
```

Output yang diharapkan:

- [ ] `models/lora/model`
- [ ] `models/lora/metrics.json`
- [ ] `models/lora/classification_report.txt`
- [ ] `models/lora/test_predictions.csv`

Kriteria selesai:

- [ ] Training selesai tanpa crash.
- [ ] `test_f1_macro` tercatat.
- [ ] Distribusi prediksi masuk akal dan tidak collapse ke satu kelas.

### 3. Train baseline full fine-tune

Tujuan: menyediakan pembanding terhadap LoRA pada dataset dan split yang setara.

Perintah:

```powershell
.\.venv\Scripts\python.exe train_baseline.py
```

Output yang diharapkan:

- [ ] `models/baseline/model`
- [ ] `models/baseline/metrics.json`
- [ ] `models/baseline/classification_report.txt`
- [ ] `models/baseline/test_predictions.csv`

Kriteria selesai:

- [ ] Training selesai tanpa crash.
- [ ] `test_f1_macro` baseline tercatat.
- [ ] Hasil siap dibandingkan dengan LoRA.

### 4. Quick comparison setelah dua training pertama

Tujuan: menentukan model mana yang dijadikan dasar untuk tahap uncertainty.

- [ ] Bandingkan `test_f1_macro` LoRA vs baseline.
- [ ] Bandingkan `test_f1_weighted`, waktu training, dan ukuran trainable parameter.
- [ ] Pilih satu model utama untuk MC Dropout.

Aturan keputusan yang disarankan:

- [ ] Jika LoRA mendekati baseline, pilih LoRA karena lebih efisien.
- [ ] Jika baseline unggul jelas pada `f1_macro`, pilih baseline sebagai kandidat utama.

### 5. Konfirmasi artefak sebelum tahap uncertainty dan evaluate

Tujuan: memastikan hasil training tersimpan di lokasi yang memang akan dibaca script lanjutan.

- [ ] Pastikan baseline menghasilkan `models/baseline/metrics.json`.
- [ ] Pastikan LoRA menghasilkan `models/lora/metrics.json`.
- [ ] Pastikan retraining nanti menghasilkan `models/retrained/metrics.json`.
- [ ] Pastikan output MC Dropout masuk ke `data/processed/uncertainty`.
- [ ] Pastikan output noise detection masuk ke `data/processed/noise`.

Kriteria selesai:

- [ ] Semua script saling kompatibel tanpa rename manual setelah dijalankan.

### 6. Jalankan MC Dropout uncertainty

Tujuan: mengukur ketidakpastian prediksi model untuk mendeteksi kandidat noisy label.

Contoh perintah jika memakai model baseline:

```powershell
.\.venv\Scripts\python.exe predict_mc_dropout.py --input_csv data/processed/dataset_absa.csv --model_dir models/baseline/model
```

Contoh perintah jika memakai model LoRA:

```powershell
.\.venv\Scripts\python.exe predict_mc_dropout.py --input_csv data/processed/dataset_absa.csv --model_dir models/lora/model
```

Output yang diharapkan:

- [ ] `data/processed/uncertainty/mc_predictions.csv`
- [ ] `data/processed/uncertainty/mc_summary.json`
- [ ] kolom entropy dan variance tersedia

Kriteria selesai:

- [ ] Nilai uncertainty berhasil dihitung untuk seluruh baris ABSA.
- [ ] File output siap dipakai oleh tahap noise detection.

### 7. Jalankan noisy label detection

Tujuan: memisahkan baris yang kemungkinan clean dari kandidat noisy.

Perintah dasar:

```powershell
.\.venv\Scripts\python.exe detect_label_noise.py
```

Output yang diharapkan:

- [ ] `data/processed/noise/clean_data.csv`
- [ ] `data/processed/noise/noisy_data.csv`
- [ ] `data/processed/noise/noise_summary.json`

Kriteria selesai:

- [ ] Rasio noisy candidate tercatat.
- [ ] Subset clean siap dipakai retraining.

### 8. Retrain pada clean subset

Tujuan: menguji apakah filtering noisy label meningkatkan performa model.

Perintah dasar:

```powershell
.\.venv\Scripts\python.exe retrain_filtered.py
```

Output yang diharapkan:

- [ ] `models/retrained/model`
- [ ] `models/retrained/metrics.json`
- [ ] `models/retrained/classification_report.txt`
- [ ] `models/retrained/test_predictions.csv`

Kriteria selesai:

- [ ] Retraining selesai tanpa crash.
- [ ] `test_f1_macro` retraining tercatat.
- [ ] Hasil siap dibandingkan dengan baseline dan LoRA.

### 9. Jalankan evaluate

Tujuan: menghasilkan ringkasan perbandingan seluruh eksperimen yang sudah dijalankan.

Perintah:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Output yang diharapkan:

- [ ] summary evaluasi JSON
- [ ] tabel perbandingan baseline vs LoRA vs retrained
- [ ] delta performa antar model

Kriteria selesai:

- [ ] Semua eksperimen terbaca oleh `evaluate.py`.
- [ ] Model terbaik dapat dipilih berdasarkan `f1_macro` dan efisiensi.

## Urutan Ringkas Eksekusi Command

```powershell
.\.venv\Scripts\python.exe train_lora.py
.\.venv\Scripts\python.exe train_baseline.py
.\.venv\Scripts\python.exe predict_mc_dropout.py --input_csv data/processed/dataset_absa.csv --model_dir models/lora/model
.\.venv\Scripts\python.exe detect_label_noise.py
.\.venv\Scripts\python.exe retrain_filtered.py
.\.venv\Scripts\python.exe evaluate.py
```

## Catatan Pengambilan Keputusan

- Fokus utama evaluasi adalah `f1_macro`, bukan accuracy, karena distribusi label sangat timpang.
- Jika resource terbatas, LoRA layak dijadikan kandidat utama lebih dulu.
- Perilaku `[SKIP]` pada `evaluate.py` normal jika suatu eksperimen belum dijalankan.

## Deliverable Minimum Sampai Evaluate

- [ ] Model baseline terlatih
- [ ] Model LoRA terlatih
- [ ] Satu eksperimen MC Dropout selesai
- [ ] Clean/noisy split tersedia
- [ ] Model retrained tersedia
- [ ] Satu file summary evaluasi final tersedia