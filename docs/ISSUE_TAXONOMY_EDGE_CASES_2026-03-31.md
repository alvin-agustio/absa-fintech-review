# Edge Cases Taxonomy

Tanggal: 2026-03-31

Catatan: referensi canonical untuk taxonomy final ada di [ISSUE_TAXONOMY_DIAMOND_STANDARD_2026-03-31.md](./ISSUE_TAXONOMY_DIAMOND_STANDARD_2026-03-31.md). Dokumen ini menjadi lampiran edge case yang menjelaskan contoh overlap dan prinsip aman.

## Tujuan

Dokumen ini merangkum edge case penting untuk taxonomy issue agar tetap aman dipakai pada standard diamond. Fokusnya bukan menambah aturan sebanyak mungkin, tetapi menjaga agar label issue tetap konsisten, mudah diaudit, dan tidak memaksa teks masuk ke bucket yang salah.

## Area Overlap

Beberapa kata memang bisa muncul di lebih dari satu aspek. Ini wajar, jadi aturan taxonomy tidak boleh memaksa satu kata hanya milik satu aspek.

Overlap yang paling perlu dijaga:

1. `Risk` vs `Trust`
   - Contoh kata rawan overlap: `data`, `privasi`, `keamanan`, `kontak`, `nomor`, `dibekukan`, `diblokir`, `freeze`, `suspend`.
   - Aturan praktis: kalau teks menyoroti bahaya, penyalahgunaan, atau keamanan data, condong ke `Trust`. Kalau yang disorot adalah konsekuensi finansial, limit, denda, atau penagihan, condong ke `Risk`.

2. `Risk` vs `Service`
   - Contoh kata rawan overlap: `approve`, `acc`, `cair`, `pencairan`, `pengajuan`, `ditolak`.
   - Aturan praktis: kalau konteksnya hasil pembiayaan atau keputusan kredit, pilih `Risk`. Kalau konteksnya alur aplikasi, fitur, atau proses operasional, pilih `Service`.

3. `Trust` vs `Service`
   - Contoh kata rawan overlap: `respon`, `balas`, `customer service`, `admin`, `tidak dibalas`.
   - Aturan praktis: kalau inti keluhannya soal kecepatan atau kualitas bantuan, pilih `Service`. Kalau inti keluhannya soal rasa aman, kejelasan, atau kepercayaan pada proses, pilih `Trust`.

## Aturan Praktis Memilih Issue

1. Pilih issue yang paling spesifik dan paling kuat muncul di teks.
2. Jika ada dua kandidat, pilih yang paling dekat dengan makna utama kalimat, bukan hanya keyword yang paling sering muncul.
3. Kalau teks berisi beberapa masalah sekaligus, tetap ambil satu issue utama saja.
4. Jika semua keyword terasa lemah atau terlalu umum, gunakan `Belum cukup spesifik`.
5. Jika frasa spesifik muncul, frasa itu harus mengalahkan keyword generik.
6. Jangan memaksa issue hanya karena ada satu kata pemicu tanpa konteks yang cukup.

## Hal Yang Sengaja Tidak Dipaksa

Taxonomy keyword tidak harus menyelesaikan semua kasus. Beberapa hal memang sengaja dibiarkan sebagai general:

1. Review sangat pendek atau spam.
2. Review yang isinya campuran dan tidak punya satu inti masalah yang jelas.
3. Kritik yang terlalu umum, misalnya hanya berisi marah tanpa konteks masalah.
4. Review dengan dua aspek yang sama-sama kuat, tetapi tidak ada tanda mana yang paling dominan.
5. Kata yang terlalu ambigu dan tidak cukup aman untuk dipetakan otomatis.

## Prinsip Diamond

Untuk standard diamond, taxonomy harus tetap:

- konsisten
- explainable
- bisa diaudit
- tidak terlalu agresif memaksa label

Kalau sebuah review masih ambigu, lebih aman dibiarkan masuk `Belum cukup spesifik` daripada dipaksa ke issue yang salah.
