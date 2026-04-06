# Issue Taxonomy Diamond Standard

Tanggal freeze: 2026-03-31

Status: canonical reference untuk dashboard issue taxonomy

Version: `diamond-standard-v1.0`

## Tujuan

Dokumen ini mengunci taxonomy issue yang dipakai di dashboard agar konsisten, explainable, dan aman untuk pemakaian "diamond standard". Fokusnya adalah mapping issue per aspek untuk `Risk`, `Trust`, dan `Service`, lalu bagaimana hasilnya dipakai pada diagnosis dan summary.

## Ruang Lingkup

Taxonomy ini hanya berlaku untuk:

1. Deteksi presence aspek.
2. Pemilihan issue utama per review pada masing-masing aspek.
3. Ringkasan naratif pada dashboard, termasuk `Diagnosis Singkat`, `Issue Map per Aspek`, dan `Summary Kesimpulan`.

Taxonomy ini tidak dipakai untuk:

1. Melatih label ABSA utama.
2. Multi-label issue extraction.
3. Analisis linguistik yang lebih berat dari keyword dan phrase matching.

## Canonical Aspects

Tiga aspek yang dikunci:

1. `Risk`
2. `Trust`
3. `Service`

## Canonical Issue Scope

### Risk

Issue bucket yang dikunci:

- `Limit, approval, dan pencairan`
- `Bunga, biaya, dan denda`
- `Penagihan dan debt collector`
- `Blokir setelah pembayaran`
- `Keamanan data pribadi`

### Trust

Issue bucket yang dikunci:

- `Transparansi dan kejelasan`
- `Legalitas, OJK, dan reputasi`
- `Penipuan dan fraud`
- `Peretasan dan keamanan akun`
- `Akun, fairness, dan suspend`
- `Privasi dan penyalahgunaan data`
- `Komunikasi dan kepastian proses`

### Service

Issue bucket yang dikunci:

- `Bug, error, dan stabilitas`
- `Pendaftaran, login, dan verifikasi`
- `CS dan respon admin`
- `Fitur, pencarian, dan katalog`
- `Proses dan pencairan`
- `Kemudahan penggunaan dan UX`
- `Performa dan stabilitas aplikasi`
- `Update dan gangguan aplikasi`

## Prinsip Mapping

Taxonomy ini mengikuti prinsip berikut:

1. Pilih issue yang paling spesifik terhadap inti kalimat.
2. Frasa spesifik harus mengalahkan keyword generik.
3. Satu review hanya memilih satu issue utama per aspek.
4. Jika sinyal terlalu lemah atau terlalu umum, pakai `Belum cukup spesifik`.
5. Mapping harus tetap mudah diaudit dari matched keywords.
6. Coverage yang bagus tidak boleh mengorbankan konsistensi.

## Alur Kerja Canonical

Alur yang dikunci di dashboard adalah:

1. Review masuk ke aspek `Risk`, `Trust`, atau `Service`.
2. Presence keyword dipakai untuk menandai apakah aspek muncul.
3. Issue rule dipilih dari frasa dan keyword yang cocok.
4. Summary merangkai hasil issue menjadi kesimpulan yang lebih tinggi tingkatannya.

## Tie-Break Rule

Jika lebih dari satu issue cocok, urutan penilaian yang dipakai adalah:

1. Phrasal specificity yang paling tinggi.
2. Total matched phrase score.
3. Rule priority.
4. Jumlah multiword hits.
5. Jumlah keyword hits.
6. Urutan taxonomy sebagai fallback terakhir.

Makna praktisnya:

- Frasa yang lebih panjang dan lebih jelas menang.
- Rule yang memang sengaja diprioritaskan akan dipilih saat sinyal setara.
- Kalau tetap imbang, urutan taxonomy menjadi penentu terakhir agar hasil tetap deterministik.

## Edge Cases Yang Dikunci

### Risk vs Trust

Gunakan `Trust` jika inti keluhannya adalah:

- privasi
- keamanan data
- penyalahgunaan data
- rasa aman
- kejelasan proses

Gunakan `Risk` jika inti keluhannya adalah:

- limit
- approval
- pencairan
- bunga
- denda
- tagihan
- penagihan

### Risk vs Service

Gunakan `Risk` jika yang dibahas adalah hasil keputusan finansial atau pembiayaan.

Gunakan `Service` jika yang dibahas adalah alur aplikasi, gangguan teknis, atau proses operasional aplikasi.

### Trust vs Service

Gunakan `Trust` jika inti masalahnya adalah rasa yakin, transparansi, kepastian, atau penyalahgunaan.

Gunakan `Service` jika inti masalahnya adalah respon bantuan, alur penggunaan, atau perilaku fitur aplikasi.

### Blocking, freeze, suspend

Aturan praktis:

- Jika blokir atau suspend dikaitkan dengan fairness, alasan yang tidak jelas, atau akses akun, condong ke `Trust`.
- Jika blokir muncul setelah pembayaran atau pelunasan, condong ke `Risk`.
- Jika masalahnya murni akun tidak bisa diakses karena bug atau update, condong ke `Service`.

## Batasan

Taxonomy ini sengaja tidak memaksa semua kasus masuk label spesifik.

1. Review yang sangat pendek tetap boleh masuk `Belum cukup spesifik`.
2. Review yang campuran boleh tetap punya satu issue utama saja.
3. Slang, typo, dan variasi ejaan tidak dijamin tertangkap seluruhnya.
4. Jika satu kata rawan overlap, konteks utama tetap lebih penting daripada kata itu sendiri.
5. Taxonomy ini bukan pengganti interpretasi manual untuk kasus yang benar-benar ambigu.

## Apa Yang Tidak Diubah Lagi

Selama freeze ini aktif, yang dianggap stabil adalah:

1. Tiga aspek utama: `Risk`, `Trust`, `Service`.
2. Label issue canonical per aspek.
3. Tie-break berbasis spesifisitas, priority, dan urutan taxonomy.
4. Guardrail `Belum cukup spesifik`.
5. Prinsip bahwa hasil dashboard harus explainable dan bisa diaudit.

## Implikasi Untuk Dashboard

Dokumen ini adalah referensi final untuk:

- diagnosis singkat per aspek
- issue map per aspek
- summary kesimpulan berbasis sinyal ABSA

Jika ada perubahan besar setelah freeze, perubahan itu harus diperlakukan sebagai revisi versi baru, bukan sekadar tweak kecil.

