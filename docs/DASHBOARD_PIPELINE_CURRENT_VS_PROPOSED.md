# Dashboard Data Flow: Current vs Proposed

Dokumen ini menjelaskan dua hal:

1. bagaimana alur dashboard **saat ini** bekerja di repo ini
2. bagaimana alur **yang diusulkan** jika dashboard diubah menjadi lebih kuat dari sisi data engineering

Tujuan dokumen ini bukan membuat desain yang terlalu rumit, tapi memberi gambaran yang **realistis, sederhana, dan bisa benar-benar dikerjakan**.

---

## 1. Gambaran Singkat

Saat ini dashboard masih memakai pola:

- user buka dashboard
- user pilih aplikasi, periode, model, dan review limit
- dashboard melakukan fetch review dari Google Play
- dashboard preprocess teks
- dashboard jalankan model
- dashboard simpan hasil sebagai cache/job lokal
- dashboard baru menampilkan insight

Pola ini cocok untuk:

- eksperimen
- demo internal
- validasi manual

Tapi pola ini kurang ideal kalau dashboard mau terasa seperti sistem yang selalu siap pakai, karena:

- user harus menunggu proses fetch dan infer
- hasil bergantung pada klik user saat itu
- job snapshot bisa menumpuk
- dashboard terasa seperti tool analisis manual, bukan observatory yang hidup

Karena itu, usulan arsitekturnya adalah:

- proses fetch, cleaning, dan inference dipindah ke pipeline terjadwal
- dashboard hanya membaca data yang sudah siap
- user tinggal pilih jendela waktu seperti:
  - 6 bulan terakhir
  - 3 bulan terakhir
  - 1 bulan terakhir
  - 2 minggu terakhir
  - 1 minggu terakhir

---

## 2. Current Flow

## 2.1 Cara kerja saat ini

Secara sederhana, alur dashboard saat ini seperti ini:

```text
User buka dashboard
  -> pilih app + periode + model + review limit
  -> app memanggil fetch Google Play
  -> review dibersihkan
  -> model dijalankan
  -> hasil disimpan ke local store/cache
  -> dashboard membaca hasil itu untuk visualisasi
```

## 2.2 Komponen utama saat ini

Beberapa file penting yang menjalankan alur sekarang:

- [app.py](/C:/Users/alvin/Downloads/skripsi/app.py)
  - UI Streamlit
  - memicu fetch dan inference dari dashboard
- [src/dashboard/live.py](/C:/Users/alvin/Downloads/skripsi/src/dashboard/live.py)
  - orchestrator fetch -> preprocess -> infer
- [src/dashboard/storage.py](/C:/Users/alvin/Downloads/skripsi/src/dashboard/storage.py)
  - menyimpan job, review cache, prediction cache
- [src/dashboard/analytics.py](/C:/Users/alvin/Downloads/skripsi/src/dashboard/analytics.py)
  - mengubah hasil prediksi menjadi KPI, trend, evidence

## 2.3 Kelebihan current flow

- sederhana untuk dipahami
- enak untuk eksperimen
- fleksibel, karena user bisa fetch scope apa saja kapan saja
- cocok saat sistem masih aktif dikembangkan

## 2.4 Kekurangan current flow

- dashboard menjadi berat karena fetch dan infer dilakukan dari UI
- user harus menunggu
- hasil tidak selalu konsisten sebagai “snapshot resmi”
- data engineering story kurang kuat karena pipeline masih sangat bergantung pada interaksi dashboard
- cache job bisa banyak, tapi tidak semuanya benar-benar berguna jangka panjang

---

## 3. Masalah Utama yang Mau Diselesaikan

Kalau dashboard mau naik kelas, masalah utamanya bukan visual saja, tetapi alur datanya.

Yang ingin diperbaiki:

- dashboard tidak perlu fetch ulang saat dibuka
- dashboard tidak perlu infer ulang saat user cuma mau melihat insight
- hasil periode populer selalu siap
- penyimpanan data menjadi lebih rapi
- ada pemisahan jelas antara:
  - ingestion
  - preprocessing
  - inference
  - serving
  - visualization

---

## 4. Proposed Flow

## 4.1 Ide utama

Dashboard diubah dari:

- **interactive fetch console**

menjadi:

- **scheduled sentiment observatory**

Artinya:

- data dikumpulkan oleh pipeline terjadwal
- data dibersihkan oleh pipeline
- model dijalankan oleh pipeline
- hasil prediksi disimpan ke tabel fakta
- dashboard tinggal query berdasarkan waktu

## 4.2 Gambaran sederhana

```text
Airflow / scheduler
  -> scrape review terbaru
  -> preprocess dan dedup
  -> jalankan model final
  -> simpan hasil ke store
  -> refresh summary

Dashboard
  -> baca data siap pakai
  -> filter berdasarkan app dan time window
  -> tampilkan insight
```

---

## 5. Proposed Architecture

## 5.1 Prinsip desain

Arsitektur yang diusulkan memakai prinsip ini:

- dashboard tidak lagi menjadi tempat utama menjalankan pipeline berat
- pipeline berat dijalankan di belakang layar secara terjadwal
- dashboard fokus pada query, agregasi, dan visualisasi
- tetap sederhana, tidak over-engineered

## 5.2 Komponen utama yang diusulkan

### A. Scheduler / Orchestrator

Contoh: Airflow

Tugasnya:

- menjalankan pipeline secara berkala
- memastikan urutan antar tahap
- mencatat status run

### B. Raw Review Store

Berisi hasil scraping mentah.

Contoh isi:

- `review_id_ext`
- `app_id`
- `app_name`
- `review_date`
- `rating`
- `review_text_raw`
- `scraped_at`
- `source_run_id`

### C. Clean Review Store

Berisi hasil preprocessing.

Contoh isi:

- `review_id_ext`
- `review_text_clean`
- `is_duplicate`
- `is_short_review`
- `cleaned_at`

### D. Prediction Fact Store

Berisi prediksi model final.

Contoh isi:

- `review_id_ext`
- `model_id`
- `aspect`
- `pred_label`
- `confidence`
- `prob_negative`
- `prob_neutral`
- `prob_positive`
- `predicted_at`

### E. Dashboard Summary Layer

Opsional, tetapi berguna.

Berisi agregasi yang sudah siap dipakai dashboard, misalnya:

- sentiment count per app per day
- aspect pressure per period
- confidence summary per period

Kalau data belum terlalu besar, summary ini bahkan bisa dihitung langsung dari fact table saat runtime.

---

## 6. Proposed DAG / Pipeline Stages

## 6.1 Stage 1: Scrape Reviews

Pipeline mengambil review terbaru dari Google Play untuk:

- Kredivo
- Akulaku

Output:

- raw review table

Catatan realistis:

- ini bukan real-time streaming
- ini adalah scheduled pull, misalnya harian

## 6.2 Stage 2: Preprocess

Pipeline membersihkan review:

- hapus URL
- hapus emoji
- lowercase
- normalisasi
- dedup
- buang review terlalu pendek

Output:

- clean review table

## 6.3 Stage 3: Inference

Pipeline menjalankan model final terhadap review yang belum diprediksi.

Output:

- prediction fact table

Catatan:

- model yang dipakai sebaiknya satu model final aktif
- jika nanti mau compare model, itu bisa dibuat sebagai mode riset, bukan mode utama dashboard

## 6.4 Stage 4: Aggregate / Serve

Pipeline bisa membuat summary harian atau periodik.

Contoh:

- jumlah review per hari
- distribusi sentimen per aspek
- top issue keywords
- confidence average

Output:

- summary table
- atau view siap query

## 6.5 Stage 5: Dashboard Read Layer

Dashboard hanya membaca data yang sudah ada.

User tinggal pilih:

- app
- time window
- compare mode

Lalu dashboard menampilkan:

- trend
- evidence
- alert
- issue cluster

---

## 7. Time Window yang Diusulkan

Window populer yang cocok untuk UI:

- 6 bulan terakhir
- 3 bulan terakhir
- 1 bulan terakhir
- 2 minggu terakhir
- 1 minggu terakhir

Catatan penting:

Window ini **tidak perlu disimpan sebagai file atau batch terpisah**.

Yang lebih baik:

- simpan semua review + prediksi ke tabel utama
- dashboard query berdasarkan `review_date`

Jadi:

- `last_7d`
- `last_14d`
- `last_30d`
- `last_90d`
- `last_180d`

adalah **hasil query**, bukan file yang berdiri sendiri.

Ini lebih rapi dan lebih scalable.

---

## 8. Current vs Proposed

## 8.1 Ringkasan perbandingan

### Current

- fetch dari dashboard
- inference dari dashboard
- cache berdasarkan klik user
- cocok untuk eksperimen
- kurang cocok untuk observatory yang selalu siap

### Proposed

- fetch lewat scheduler
- inference lewat scheduler
- dashboard hanya baca hasil
- lebih cepat untuk user
- lebih kuat untuk sisi data engineering

## 8.2 Tabel singkat

| Area | Current | Proposed |
|---|---|---|
| Trigger pipeline | Klik user di dashboard | Scheduler terjadwal |
| Fetch data | Saat user meminta | Otomatis di belakang layar |
| Inference | Saat user meminta | Otomatis setelah preprocessing |
| Storage | Job cache per scope | Canonical raw/clean/prediction tables |
| Dashboard role | Tool analisis aktif | Read-only observatory |
| UX | Tunggu proses | Hasil langsung siap |
| Data engineering story | Cukup | Jauh lebih kuat |

---

## 9. Apakah Airflow Realistis?

Jawaban jujurnya: **iya, tapi dengan batas yang sehat**.

Airflow realistis kalau tujuannya:

- memperkuat sisi data engineering
- menunjukkan pipeline orchestration
- membuat dashboard lebih siap demo

Tapi tetap harus realistis:

- sumber data tetap Google Play
- bukan streaming real-time
- update kemungkinan harian atau beberapa kali per hari
- deployment tetap lebih berat dibanding scheduler sederhana

Jadi istilah yang lebih tepat adalah:

- **near-real-time scheduled pipeline**

bukan:

- real-time streaming system

---

## 10. Rekomendasi Implementasi Bertahap

Supaya realistis, implementasinya tidak perlu langsung “production-grade penuh”.

### Tahap 1

Pisahkan mode dashboard menjadi:

- **Observatory mode**
  - hanya baca data siap pakai
- **Lab mode**
  - tetap boleh ada fetch manual untuk eksperimen

Ini langkah paling aman.

### Tahap 2

Buat pipeline terjadwal sederhana untuk:

- scrape raw review
- preprocess
- infer model final
- simpan hasil

### Tahap 3

Dashboard dipindah untuk membaca:

- raw facts
- prediction facts
- summary by period

### Tahap 4

Kalau sudah stabil, barulah:

- hapus ketergantungan fetch manual dari halaman utama
- pertahankan fetch manual hanya di mode admin/lab

---

## 11. Recommended Final Shape

Kalau tujuannya adalah dashboard skripsi + portfolio yang kuat, bentuk akhir yang saya sarankan:

### Mode utama

Dashboard utama hanya menampilkan:

- 6 bulan terakhir
- 3 bulan terakhir
- 1 bulan terakhir
- 2 minggu terakhir
- 1 minggu terakhir

dengan filter:

- Kredivo
- Akulaku
- Both

### Mode tambahan

Halaman atau tab khusus untuk:

- manual fetch
- debug pipeline
- compare model
- gold subset / research lab

Dengan cara ini:

- user umum melihat dashboard yang cepat dan rapi
- user teknis tetap punya ruang eksperimen

---

## 12. Kesimpulan

Current flow di repo ini masih sangat berguna untuk eksplorasi dan eksperimen.

Tetapi kalau dashboard ingin tampil sebagai sistem yang lebih matang, lebih profesional, dan lebih kuat dari sisi data engineering, maka arsitektur yang lebih tepat adalah:

- pipeline terjadwal untuk fetch + preprocess + inference
- dashboard hanya membaca data siap pakai
- time window menjadi query, bukan batch manual

Kesimpulan paling sederhana:

- **current flow cocok untuk eksperimen**
- **proposed flow cocok untuk observatory**

Dan untuk repo ini, pendekatan paling realistis adalah:

- **jangan buang flow lama**
- tetapi pindahkan flow lama ke **Lab/Admin mode**
- lalu bangun **Observatory mode** di atas scheduled pipeline

Itu memberi hasil terbaik dari dua dunia:

- tetap fleksibel untuk riset
- tetapi jauh lebih kuat untuk demo, presentasi, dan data engineering narrative
