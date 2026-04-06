# Insight Layer Validation

Tanggal: 2026-03-31

Dokumen ini mengunci cara validasi ringan untuk layer insight dashboard setelah taxonomy dianggap freeze. Fokusnya bukan mengevaluasi model ABSA dari nol, tetapi mengecek apakah `Diagnosis Singkat`, `Issue Map per Aspek`, dan `Summary Kesimpulan` masih selaras dengan sinyal data yang sebenarnya.

## Tujuan

Validasi ini dipakai untuk menjawab tiga pertanyaan sederhana:

1. Apakah summary menunjuk aspek terbaik dan terburuk yang benar.
2. Apakah issue utama yang dibawa ke summary masih konsisten dengan breakdown issue.
3. Apakah output yang tampil di dashboard masih masuk akal saat dicek manual terhadap contoh review.

## Artefak Audit

Script yang dipakai:

- `scripts/audit_insight_layer.py`

Output default:

- `data/processed/audits/insight_layer/insight_layer_report.json`
- `data/processed/audits/insight_layer/summary_score_table.csv`
- `data/processed/audits/insight_layer/issue_snapshot.csv`
- `data/processed/audits/insight_layer/app_summary_audit.csv`
- `data/processed/audits/insight_layer/manual_review_examples.csv`

## Ringkasan Hasil Saat Ini

Audit dijalankan pada:

- `data/processed/dataset_absa_50k_v2_intersection.csv`

Ringkasan utama:

1. Status summary: `ready`
2. Quality summary: `high`
3. Focus aspect yang dipilih summary: `Risk`
4. Focus aspect dari distribusi negatif: `Risk`
5. Best aspect yang dipilih summary: `Service`
6. Best aspect dari distribusi positif: `Service`
7. App cards tersedia untuk `Kredivo` dan `Akulaku`

Artinya, mapping tingkat atas dari rule-based summary sudah konsisten dengan distribusi sentimen yang menjadi dasarnya.

## Temuan Penting

1. `Risk` sudah paling stabil sebagai sinyal negatif utama.
   - Top issue: `Limit, approval, dan pencairan`
   - Share issue utama: `51.5%`
   - Generic share: `19.0%`

2. `Trust` masih belum sekuat `Risk`.
   - Top issue masih `Belum cukup spesifik`
   - Generic share: `46.1%`

3. `Service` masih jadi area insight yang paling lemah untuk issue-level summary.
   - Top issue masih `Belum cukup spesifik`
   - Generic share: `54.8%`

4. Secara ringkas, summary level sudah stabil, tetapi issue elaboration untuk `Trust` dan terutama `Service` masih perlu dibaca hati-hati.

## Cara Membaca Hasil Audit

### 1. Summary score table

Dipakai untuk memastikan:

- aspek dengan `positive_share` tertinggi memang menjadi sisi paling kuat
- aspek dengan `negative_share` tertinggi memang menjadi fokus perhatian

### 2. Issue snapshot

Dipakai untuk memastikan:

- issue yang dibawa ke summary memang punya dasar yang cukup kuat
- generic share belum terlalu besar untuk aspek yang ingin dielaborasi

### 3. App summary audit

Dipakai untuk memastikan:

- narasi `Kredivo` dan `Akulaku` tidak tertukar
- app-level summary tetap konsisten dengan distribusi sentimen per app

### 4. Manual review examples

Dipakai untuk audit cepat oleh manusia:

- baca contoh `Positive`, `Neutral`, dan `Negative` per aspek
- baca contoh `Negative` per app dan per aspek
- cek apakah narasi dashboard terasa terlalu keras, terlalu lemah, atau sudah pas

## Guardrail Praktis

Summary dianggap masih aman dipakai jika:

1. Focus aspect summary sama dengan aspek dengan `negative_share` tertinggi.
2. Best aspect summary sama dengan aspek dengan `positive_share` tertinggi.
3. App cards muncul untuk semua app utama yang memang ada di scope.
4. Issue utama yang disebut bukan hasil dari bucket yang terlalu generik tanpa konteks.

Kalau `generic share` untuk suatu aspek masih sangat tinggi, summary tetap boleh menyebut aspek itu, tetapi elaborasi issue spesifiknya harus ditulis lebih hati-hati.

## Putusan Praktis

Untuk kondisi sekarang:

1. Rule-based summary sudah cukup stabil untuk dashboard.
2. `Risk` sudah cukup aman untuk dijadikan sinyal inti.
3. `Trust` dan `Service` masih layak ditampilkan, tetapi jangan terlalu agresif saat menyebut issue detail.
4. Jika ada iterasi berikutnya, audit insight paling bernilai tetap ada pada issue coverage `Service`, bukan pada rewrite total summary.
