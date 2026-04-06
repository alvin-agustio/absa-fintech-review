# Issue Taxonomy Audit

Tanggal: 2026-03-31

Catatan: dokumen canonical yang mengunci taxonomy final ada di [ISSUE_TAXONOMY_DIAMOND_STANDARD_2026-03-31.md](./ISSUE_TAXONOMY_DIAMOND_STANDARD_2026-03-31.md). Dokumen ini dipertahankan sebagai ringkasan audit dan latar belakang keputusan.

## Ringkasan

Mapping per aspek saat ini sudah cukup kuat untuk dashboard dan penulisan kesimpulan, tetapi belum bisa disebut maksimal. Alurnya sudah jelas: review masuk ke aspek `Risk`, `Trust`, atau `Service`, lalu issue dipilih dengan rule keyword, lalu hasilnya dirangkai lagi ke summary. Itu sudah bagus untuk sistem yang explainable, tapi masih ada ruang perbaikan di coverage dan ketelitian issue.

## Limitasi Saat Ini

1. Banyak issue masih berbasis keyword manual, jadi variasi slang, typo, dan frasa tidak langsung kadang belum tertangkap.
2. Sebagian review masih jatuh ke label `Belum cukup spesifik`, artinya taxonomy belum menangkap semua pola yang muncul.
3. Satu review kadang punya lebih dari satu masalah, tetapi sistem sekarang hanya memilih issue utama.
4. Ada potensi overlap antar aspek, misalnya kata tentang `data`, `proses`, atau `approval` bisa nyerempet lebih dari satu aspek.
5. Mapping saat ini lebih kuat untuk keluhan negatif daripada untuk elaborasi sisi positif.

## Apakah Corpus Perlu Diperluas

Corpus yang ada sudah besar dan layak:
- `reviews_clean_v2.csv` sekitar 270 ribu review
- `dataset_absa_50k_v2_intersection.csv` sekitar 48 ribu review

Jadi untuk tahap sekarang, yang paling penting bukan langsung memperbesar corpus untuk training, tetapi memperkaya audit taxonomy dari corpus yang sudah ada. Corpus lebih luas masih berguna untuk:
- mencari keyword baru
- melihat overlap antar aspek
- menemukan review yang masih masuk bucket umum

Kesimpulan singkat: corpus belum harus diperluas dulu, tapi audit taxonomy masih sangat layak diperluas.

## Urutan Kerja Paling Efektif

1. Audit dulu semua review yang masih masuk `Belum cukup spesifik`.
2. Tambah keyword dan frasa baru untuk issue yang paling sering muncul.
3. Cek overlap antar aspek supaya istilah yang mirip tidak saling tabrakan.
4. Rapikan issue yang terlalu umum, lalu pecah bila memang ada subpola yang jelas.
5. Setelah taxonomy cukup stabil, baru putuskan apakah corpus perlu perluasan tambahan untuk kasus yang masih lemah.

## Putusan Praktis

Untuk kondisi sekarang, mapping per aspek sudah cukup untuk dipakai, tetapi belum maksimal. Fokus terbaik adalah memperkuat taxonomy issue dulu, bukan langsung memperbesar corpus. Kalau setelah audit masih banyak bucket umum, barulah perluasan corpus jadi langkah berikutnya.
