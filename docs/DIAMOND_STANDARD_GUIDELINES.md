# Diamond Standard Guidelines for ABSA Labeling

## Purpose

Dokumen ini menjadi pedoman formal anotasi untuk Aspect-Based Sentiment Analysis pada ulasan aplikasi fintech lending Indonesia, khususnya Kredivo dan Akulaku. Tujuannya adalah memastikan proses pelabelan konsisten, dapat diaudit, dan cukup kuat untuk digunakan sebagai diamond standard guideline dalam pipeline LLM-assisted annotation.

Dokumen ini dipakai untuk:

1. acuan anotasi manual
2. acuan prompt LLM
3. acuan audit dan adjudication
4. acuan penulisan metodologi skripsi

## Unit of Analysis

Satu unit anotasi adalah satu review.

Setiap review dinilai terhadap tiga aspek secara independen:

1. Risk
2. Trust
3. Service

Satu review dapat:

1. membahas satu aspek saja
2. membahas beberapa aspek sekaligus
3. tidak membahas aspek tertentu sama sekali

## Output Label Schema

Untuk setiap review, anotator hanya menghasilkan kolom berikut:

1. `risk_sentiment`
2. `trust_sentiment`
3. `service_sentiment`
4. `reasoning`

Nilai yang valid untuk tiga kolom sentiment adalah:

1. `Positive`
2. `Negative`
3. `Neutral`
4. `null`

Makna `null`:

Aspek tersebut tidak dibahas dalam review.

Kolom `aspect_type` tidak perlu dianotasi langsung karena dapat diturunkan dari tiga kolom sentiment.

## Aspect Definitions

### 1. Risk

Aspek Risk mencakup persepsi pengguna terhadap risiko finansial dan risiko operasional yang langsung terkait pinjaman.

Contoh topik:

1. bunga
2. denda
3. penagihan
4. debt collector atau DC
5. ancaman dan teror
6. cicilan
7. tenor
8. limit kredit
9. skor kredit
10. pencairan dana
11. keamanan data pribadi
12. potongan saldo
13. risiko keterlambatan bayar

### 2. Trust

Aspek Trust mencakup persepsi pengguna terhadap kredibilitas, kejujuran, legalitas, dan reputasi platform.

Contoh topik:

1. legalitas OJK
2. transparansi biaya
3. kejujuran aplikasi
4. reputasi perusahaan
5. penipuan
6. scam
7. fraud
8. manipulasi sistem
9. rasa aman terhadap institusi

### 3. Service

Aspek Service mencakup kualitas layanan, kualitas aplikasi, dan kualitas interaksi pengguna dengan sistem atau dukungan pelanggan.

Contoh topik:

1. customer service
2. respons keluhan
3. kecepatan verifikasi
4. login
5. UI atau UX
6. bug
7. error teknis
8. kemudahan penggunaan
9. stabilitas aplikasi
10. pengalaman penggunaan fitur

## Sentiment Definitions

### Positive

Gunakan `Positive` jika pengguna secara eksplisit memuji, merasa puas, merasa terbantu, atau diuntungkan oleh aspek tersebut.

Contoh:

1. proses cepat
2. bunga kecil dan membantu
3. aplikasi mudah dipakai
4. CS responsif

### Negative

Gunakan `Negative` jika pengguna mengeluh, marah, kecewa, merasa dirugikan, merasa diperlakukan tidak adil, atau memberikan kritik tajam terhadap aspek tersebut.

Contoh:

1. bunga mencekik
2. DC kasar
3. aplikasi error terus
4. penjelasan biaya tidak jelas

### Neutral

Gunakan `Neutral` hanya jika pengguna membahas aspek tersebut secara faktual tanpa pujian atau keluhan yang jelas.

Contoh:

1. pengguna bertanya soal tenor
2. pengguna memberi saran objektif
3. pengguna menyebut fitur tanpa evaluasi emosional

### Null

Gunakan `null` jika aspek tidak dibahas sama sekali.

Catatan penting:

`Neutral` bukan pengganti `null`. Jika aspek tidak muncul, isi `null`.

## Core Annotation Rules

1. Fokus pada isi teks review, bukan pada rating bintang semata.
2. Satu review harus dievaluasi untuk ketiga aspek secara terpisah.
3. Satu review boleh memiliki lebih dari satu label sentiment jika membahas beberapa aspek.
4. Jika aspek tidak dibahas, isi `null`.
5. Jika review singkat tetapi maknanya jelas, tetap beri label sesuai makna.
6. Jika ada typo, slang, atau bahasa informal, anotator harus menafsirkan makna yang paling wajar dalam konteks Indonesia.
7. Reasoning harus observasional, singkat, dan tidak normatif.

## Decision Rules for Difficult Cases

### Sarcasm and Irony

Jika kalimat tampak seperti pujian tetapi konteksnya menyindir, gunakan makna yang sebenarnya.

Contoh:

`hebat banget aplikasinya, baru telat sehari langsung diteror`

Label yang benar:

1. Risk = `Negative`
2. Service dapat `Negative` jika konteks menyasar petugas/layanan

### Mixed Sentiment in One Aspect

Jika satu aspek mengandung pujian dan keluhan sekaligus, pilih sentimen yang paling dominan atau paling eksplisit.

Contoh:

`awalnya bagus, tapi sekarang bunganya makin tidak masuk akal`

Label:

1. Risk = `Negative`

### Mixed Aspects in One Review

Jika review membahas beberapa aspek, semua aspek yang relevan harus diisi.

Contoh:

`UI gampang dipakai tapi denda telatnya parah`

Label:

1. Service = `Positive`
2. Risk = `Negative`
3. Trust = `null`

### Question-Only Reviews

Jika review hanya bertanya tanpa evaluasi emosional yang jelas, gunakan `Neutral` pada aspek yang benar-benar dibahas.

Contoh:

`kalau telat sehari kena denda berapa ya?`

Label:

1. Risk = `Neutral`

### Complaint Without Explicit Aspect Name

Jika nama aspek tidak disebut langsung, tetapi implikasinya jelas, anotator tetap harus memberi label.

Contoh:

`aplikasi suka nge-freeze pas verifikasi`

Label:

1. Service = `Negative`

### Moral or Legal Judgment

Anotator tidak menilai apakah pengguna benar atau salah secara hukum. Fokusnya adalah persepsi pengguna terhadap aspek.

Contoh:

`saya ditolak terus, padahal selalu bayar tepat waktu`

Jika nada utama adalah keluhan soal limit atau kelayakan pinjaman, maka:

1. Risk = `Negative`

## Slang and Informal Language Rules

Beberapa istilah informal yang harus dipahami secara konsisten:

1. `bapuk` cenderung `Negative`
2. `scam`, `penipuan`, `gimmick` cenderung `Trust = Negative`
3. `dc galak`, `diteror`, `diancam` cenderung `Risk = Negative`
4. `lemot`, `error`, `ngelag`, `force close` cenderung `Service = Negative`
5. `mudah`, `cepat`, `membantu` cenderung `Positive` sesuai aspek terkait

Jika konteks tidak cukup jelas, anotator tetap harus mengikuti makna dominan yang paling masuk akal.

## Reasoning Rules

Kolom `reasoning` harus:

1. maksimal 1 sampai 2 kalimat
2. menjelaskan bukti tekstual utama
3. tidak mengulang definisi aspek secara generik
4. tidak memuat opini anotator di luar isi review

Contoh reasoning yang baik:

`Pengguna mengeluhkan aplikasi sulit dipakai dan prosesnya ribet, sehingga Service bernilai Negative.`

Contoh reasoning yang buruk:

`Menurut saya aplikasi ini memang jelek dan tidak layak dipakai.`

## Annotation Workflow

Urutan anotasi yang disarankan:

1. baca seluruh review sekali
2. tentukan apakah ada sinyal untuk Risk
3. tentukan apakah ada sinyal untuk Trust
4. tentukan apakah ada sinyal untuk Service
5. isi sentiment per aspek
6. isi reasoning singkat
7. cek ulang apakah ada aspek yang seharusnya `null`

## Quality Control Rules

Sebelum anotasi dianggap valid, harus lolos pengecekan berikut:

1. semua review_id dari input muncul tepat satu kali
2. semua label hanya memakai `Positive`, `Negative`, `Neutral`, atau `null`
3. tidak ada aspek yang diisi `Neutral` jika sebenarnya tidak dibahas
4. reasoning tidak kosong
5. output tidak menambah field di luar schema

## Adjudication Rules

Jika ada konflik antar anotator, manusia atau LLM, gunakan urutan adjudication berikut:

1. cek apakah aspek memang dibahas atau seharusnya `null`
2. cek bukti tekstual paling eksplisit
3. prioritaskan konteks kalimat utama dibanding kata tunggal
4. prioritaskan makna aktual dibanding rating bintang
5. jika masih ambigu, pilih label yang paling konservatif berdasarkan bukti teks

Jika bukti terlalu lemah untuk menyatakan sentimen jelas, gunakan `Neutral` hanya bila aspek memang dibahas. Jika tidak dibahas, gunakan `null`.

## Sampling Recommendation for Diamond Subset

Jika membentuk subset diamond dari `reviews_clean.csv`, metode yang direkomendasikan adalah stratified random sampling proportional berdasarkan:

1. `app_name`
2. `rating`

Strata final menjadi `app_name × rating`, sehingga struktur sampel mengikuti distribusi populasi asli.

## Minimal Output Example

```json
[
  {
    "review_id": 123,
    "risk_sentiment": "Negative",
    "trust_sentiment": null,
    "service_sentiment": "Negative",
    "reasoning": "Pengguna mengeluhkan denda dan juga menyebut aplikasi error saat digunakan."
  }
]
```

## Final Principle

Diamond standard tidak berarti anotasi harus sempurna tanpa ambigu. Diamond standard berarti aturan keputusan konsisten, terdokumentasi, dapat diaudit, dan diterapkan secara disiplin pada seluruh data.