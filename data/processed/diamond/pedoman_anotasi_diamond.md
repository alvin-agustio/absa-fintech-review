# Pedoman Anotasi Diamond Subset (ABSA)

## 1) Ukuran Diamond Subset (Final)

Ukuran yang digunakan: **300 review → ~371 unit anotasi aspect-level**.

Alasan praktis:
- 300 review realistis dianotasi sendiri dalam 4-6 jam kerja fokus.
- Setiap review disertakan semua aspeknya agar tidak ada bias evaluasi parsial.
- Total ~371 item aspect-level cukup kuat untuk evaluasi gold standard skripsi.

Komposisi:
- 300 review unik, disampling terstratifikasi berdasarkan `app_name` dan `rating_group`.
- Aspek: `risk`, `trust`, `service`.
- Setiap review mengikutsertakan semua aspek yang berlabel di silver dataset.
- Distribusi sentimen per aspek mengikuti distribusi natural silver (tidak dipaksakan seimbang).

Catatan:
- Unit anotasi adalah pasangan `(review, aspect)`, bukan hanya review.
- Kelas `Neutral` memang jarang di dataset ini karena sifat ulasan pengguna. Ini wajar dan bukan masalah.

## 2) Definisi Label

Gunakan 3 label berikut:
- `Negative`: opini terhadap aspek bernada keluhan, ketidakpuasan, risiko, kerugian, atau masalah.
- `Neutral`: deskriptif/faktual, ambigu, campuran seimbang, atau tidak cukup kuat ke positif/negatif.
- `Positive`: opini terhadap aspek bernada puas, aman, membantu, cepat, atau menguntungkan.

## 3) Aturan Inti Penilaian

1. Nilai per aspek, bukan sentimen umum review.
2. Jika aspek tidak dibahas sama sekali, isi `aspect_present = 0` dan `label = Neutral`.
3. Jika ada kata positif dan negatif sekaligus pada aspek yang sama:
   - pilih sentimen yang lebih dominan,
   - jika benar-benar seimbang, pilih `Neutral`.
4. Jangan menilai niat penulis di luar teks.
5. Jangan gunakan label LLM sebagai acuan utama. LLM hanya referensi awal.

## 4) Definisi Operasional Per Aspek

### A. Risk
Fokus pada biaya, bunga, denda, penagihan, ancaman, keamanan finansial, dan risiko penyalahgunaan.

Contoh arah:
- "Bunga mencekik" -> `Negative`
- "Biaya jelas dan sesuai" -> `Positive`
- "Ada biaya admin" (tanpa penilaian) -> `Neutral`

### B. Trust
Fokus pada kepercayaan, legalitas, transparansi, reputasi, privasi data, dan rasa aman terhadap platform.

Contoh arah:
- "Data saya disalahgunakan" -> `Negative`
- "Aplikasi terpercaya, proses jelas" -> `Positive`
- "Sudah terdaftar" (faktual tanpa evaluasi) -> `Neutral`

### C. Service
Fokus pada kualitas layanan, CS, kecepatan respon, stabilitas aplikasi, UX, bug, error, dan kemudahan penggunaan.

Contoh arah:
- "Aplikasi sering crash" -> `Negative`
- "CS cepat membantu" -> `Positive`
- "Ada fitur chat" (faktual) -> `Neutral`

## 5) Kasus Sulit (Decision Rules)

1. Sarkasme:
- Label berdasarkan makna sebenarnya, bukan kata literal.

2. Multi-kalimat bertentangan:
- Prioritaskan kalimat yang paling eksplisit terhadap aspek.

3. Menyebut brand/pihak lain:
- Tetap nilai hanya jika ada dampak ke aspek pada aplikasi target.

4. Spam/teks sangat pendek:
- Jika aspek tidak jelas -> `aspect_present = 0`, `label = Neutral`.

## 6) Prosedur Kerja Annotator (Tunggal)

1. Buka `template_anotator_tunggal.csv`.
2. Untuk setiap baris, baca `review_text` dan perhatikan kolom `aspect` yang sedang dinilai.
3. Isi kolom `label`, `aspect_present`, `confidence`, dan `notes` (jika perlu).
4. Jangan gunakan `llm_label` sebagai acuan utama — itu hanya referensi awal LLM, bukan kebenaran.
5. Setelah selesai semua, lakukan 2nd-pass ringan: review ulang ~30 item acak untuk memastikan konsistensi penilaian Anda dari awal sampai akhir.

## 7) Quality Control (Annotator Tunggal)

- Lakukan 2nd-pass: baca ulang ~10% item secara acak setelah selesai semua.
- Cek apakah label awal dan label ulang konsisten.
- Jika banyak perubahan di 2nd-pass, pertimbangkan review ulang semua item aspek yang sama.
- Catat kasus ambigu di kolom `notes` supaya keputusan Anda bisa diaudit.

## 8) Kolom yang Wajib Diisi

- `aspect_present`: 1 jika aspek dibahas, 0 jika tidak.
- `label`: Negative/Neutral/Positive.
- `confidence`: 1-3 (1=ragu, 2=sedang, 3=yakin).
- `notes`: opsional singkat untuk kasus ambigu.

## 9) Output yang Dihasilkan

- `template_anotator_tunggal.csv` — file kerja anotasi (isi kolom label, aspect_present, confidence, notes).
- `gold_final.csv` — hasil anotasi final setelah 2nd-pass selesai (simpan sebagai salinan baru).
- Ringkasan distribusi label manual vs llm_label untuk pelaporan skripsi.
