# Definition of Aspects and Sentiment Labels in Fintech Lending ABSA

Dokumen ini merangkum **definisi aspek** dan **definisi label sentimen** untuk task ABSA pada domain fintech lending di repo ini.

Dokumen ini disusun dari dua sumber aturan yang memang ada di repo:

- [labeling.py](/C:/Users/alvin/Downloads/skripsi/src/data/labeling.py)
  - dipakai untuk **weak labeling berbasis LLM**
- [pedoman_anotasi_diamond.md](/C:/Users/alvin/Downloads/skripsi/data/processed/diamond/pedoman_anotasi_diamond.md)
  - dipakai untuk **anotasi diamond/gold subset**

Tujuan dokumen ini adalah membuat satu acuan yang lebih rapi dan konsisten untuk paper, sehingga reviewer bisa melihat:

- apa arti setiap aspek
- kapan suatu review dianggap membahas aspek tertentu
- contoh sinyal tekstualnya
- bagaimana skema label sentimen dipakai

---

## Harmonized Ground Truth

Secara praktik, repo ini punya dua lapisan aturan:

1. **weak-label rules** di prompt LLM
2. **diamond/manual rules** di pedoman anotasi

Keduanya sangat mirip, tetapi tidak selalu identik pada detail kecil.

Karena itu, dokumen ini memakai **harmonized operational definition**, artinya:

- definisi umum mengikuti dua sumber tadi
- jika ada area overlap, dipilih interpretasi yang paling stabil untuk anotasi dan evaluasi

Catatan penting:

- untuk **weak labeling**, aspek yang tidak dibahas diisi `null`
- untuk **gold subset final**, aspek yang tidak dibahas direpresentasikan sebagai:
  - `aspect_present = 0`
  - `label = kosong`

Jadi, secara operasional:

- **jika aspek tidak hadir, jangan pakai label sentimen**

---

## Table 1. Definition of Aspects in Fintech Lending Domain

| Aspect | Description | What is included | Boundary / Not the main focus | Example textual signals |
|---|---|---|---|---|
| **Risk** | Persepsi pengguna terhadap risiko finansial, beban biaya, penalti, penagihan, limit, pencairan, dan konsekuensi ekonomi dari penggunaan layanan. | bunga, denda, biaya admin, cicilan, tenor, limit kredit, pencairan dana, penagihan, debt collector, ancaman, keterlambatan bayar, tagihan tetap muncul, pelunasan tapi limit tidak bisa dipakai | Jika inti keluhan adalah bug aplikasi, CS, login, verifikasi, atau UX, itu lebih dekat ke `service`. Jika inti keluhan adalah rasa tidak percaya, scam, transparansi, atau privasi data, itu lebih dekat ke `trust`. | `bunga tinggi`, `denda besar`, `dc kasar`, `tagihan masih jalan`, `limit tidak bisa dipakai`, `pengajuan pinjaman ditolak`, `pencairan tidak masuk`, `bayar tepat waktu tapi dibekukan` |
| **Trust** | Persepsi pengguna terhadap kepercayaan, transparansi, reputasi, legalitas, kejujuran, rasa aman, dan integritas platform. | aplikasi dianggap menipu, scam, fraud, tidak transparan, tidak jelas, reputasi buruk, legalitas/OJK, kekhawatiran data disalahgunakan, rasa aman terhadap institusi, sistem dianggap tidak adil atau menyesatkan | Jika inti pembahasan adalah bunga, denda, limit, penagihan, atau risiko ekonomi langsung, itu lebih dekat ke `risk`. Jika inti masalah adalah kualitas layanan atau error teknis, itu lebih dekat ke `service`. | `penipuan`, `scam`, `tidak jelas`, `menyesatkan`, `data saya disalahgunakan`, `hapus data saya`, `takut data bocor`, `platform tidak bisa dipercaya`, `tanpa alasan yang jelas akun diblokir` |
| **Service** | Persepsi pengguna terhadap kualitas layanan, pengalaman penggunaan aplikasi, respons sistem, proses verifikasi, customer service, dan aspek teknis aplikasi. | bug, error, crash, lemot, login gagal, reset PIN gagal, verifikasi lama, pengajuan selalu gagal karena proses, respons CS buruk, UI/UX membingungkan, fitur tidak berfungsi, aplikasi tidak bisa dipakai | Jika fokus utamanya adalah kerugian finansial atau kebijakan biaya, itu lebih dekat ke `risk`. Jika fokus utamanya adalah rasa ditipu, reputasi, atau privasi/kepercayaan, itu lebih dekat ke `trust`. | `aplikasi error`, `susah login`, `pengajuan gagal terus`, `akun diblokir`, `fitur tidak berfungsi`, `CS tidak membantu`, `verifikasi lama`, `tidak bisa install`, `paylater tidak bisa dipakai` |

---

## Practical Boundary Rules Between Aspects

Beberapa review di domain ini memang sering multi-aspek. Karena itu, berikut aturan pembatas yang paling aman:

### 1. Risk vs Trust

- Pakai **Risk** jika inti teks membahas:
  - bunga
  - denda
  - limit
  - penagihan
  - tagihan
  - pencairan
  - kerugian finansial langsung
- Pakai **Trust** jika inti teks membahas:
  - penipuan
  - transparansi
  - legalitas
  - reputasi
  - rasa aman
  - data/privasi
  - ketidakpercayaan pada platform

### 2. Trust vs Service

- Pakai **Service** jika masalahnya ada pada:
  - aplikasi
  - bug
  - login
  - verifikasi
  - CS
  - proses teknis
- Pakai **Trust** jika masalahnya bergeser ke:
  - platform dianggap tidak jujur
  - alasan pemblokiran tidak jelas
  - pengguna merasa ditipu
  - data dikhawatirkan disalahgunakan

### 3. Risk vs Service

- Pakai **Risk** jika isi keluhan terkait konsekuensi finansial atau approval kredit
- Pakai **Service** jika isi keluhan terkait pengalaman proses atau fitur aplikasi

Contoh:

- `limit tidak bisa dipakai` bisa condong ke `risk`
- `fitur paylater error` bisa condong ke `service`

### 4. Data and Privacy Cases

Kasus data pribadi adalah area overlap paling jelas antara dua sumber aturan.

Agar konsisten, aturan operasional yang disarankan:

- gunakan **Trust** jika inti teks adalah:
  - takut data disalahgunakan
  - minta data dihapus
  - merasa platform tidak aman
- gunakan **Risk** hanya jika data/privacy dibahas sebagai bagian dari ancaman atau risiko yang sangat terkait dengan konsekuensi finansial / penagihan

Jadi secara praktik, **privacy/data misuse lebih aman dipetakan ke Trust**.

---

## Table 2. Sentiment Label Definition

| Sentiment Label | Core meaning | When to use | Typical cues | When not to use |
|---|---|---|---|---|
| **Positive** | Pengguna memuji, merasa puas, terbantu, aman, cepat, mudah, atau diuntungkan pada aspek tertentu. | Saat aspek target jelas dibahas dan evaluasinya menguntungkan | `bagus`, `membantu`, `aman`, `mudah`, `cepat`, `memuaskan`, `terpercaya`, `bunga rendah`, `proses lancar`, `cs membantu` | Jangan pakai jika pujian hanya umum tetapi aspek target tidak hadir jelas |
| **Negative** | Pengguna mengeluh, kecewa, merasa dirugikan, marah, atau mengkritik aspek tertentu secara tegas. | Saat aspek target jelas dibahas dan nada dominannya merugikan / bermasalah | `kecewa`, `parah`, `ditolak`, `diblokir`, `bunga tinggi`, `denda`, `scam`, `penipuan`, `error`, `lemot`, `cs tidak membantu`, `dc kasar` | Jangan pakai hanya karena rating bintang rendah jika teks tidak mendukung |
| **Neutral** | Pengguna membahas aspek secara deskriptif, ambigu, faktual, atau campuran yang tidak cukup dominan ke positif/negatif. | Saat aspek target benar-benar hadir, tetapi tidak ada evaluasi kuat atau ada campuran seimbang | `ada biaya admin`, `sudah terdaftar`, `fitur tersedia`, pertanyaan objektif, informasi faktual, mixed sentiment yang benar-benar seimbang | Jangan pakai untuk menggantikan aspek yang tidak hadir |
| **No sentiment label / null / kosong** | Aspek target tidak dibahas | Saat review tidak menyentuh aspek tersebut | tidak ada sinyal relevan untuk aspek target | Jangan diubah menjadi `Neutral` hanya supaya semua aspek punya label |

---

## Label Scheme Used in This Repo

Skema label sentimen yang dipakai di repo ini adalah:

### Pada weak-labeling

- `Positive`
- `Negative`
- `Neutral`
- `null` untuk aspek yang tidak dibahas

### Pada gold subset final

- `aspect_present = 1` lalu `label` diisi salah satu:
  - `Positive`
  - `Negative`
  - `Neutral`
- `aspect_present = 0` lalu:
  - `label = kosong`

Jadi untuk paper, formulasi yang paling aman adalah:

- **task utama adalah sentiment classification pada aspek yang hadir**
- aspek yang tidak hadir **tidak diberi label sentimen**

---

## Operational Notes for Difficult Cases

### 1. Mixed sentiment

Jika satu aspek memuat pujian dan keluhan sekaligus:

- pilih yang paling dominan atau paling eksplisit
- jika benar-benar seimbang, pilih `Neutral`

### 2. Question / speculation / uncertainty

Jika review hanya berupa pertanyaan atau kebingungan:

- pakai `Neutral` hanya jika aspek target memang sedang dibahas
- jika aspek target tidak benar-benar hadir, jangan beri label

### 3. Sarcasm

Gunakan makna sebenarnya, bukan kata literal.

### 4. Multi-aspect review

Satu review bisa memuat banyak aspek.

Karena itu:

- penilaian harus selalu **per aspek**
- jangan pakai sentimen umum review sebagai label semua aspek

### 5. Rating vs text conflict

Fokus pada teks, bukan rating.

Rating hanya metadata tambahan.

---

## Short Version for Paper

Kalau mau ditulis singkat di paper:

> The dataset uses three domain-specific ABSA aspects, namely **risk**, **trust**, and **service**. Risk covers financial burden and exposure such as interest, penalties, billing, credit limit, and loan approval outcomes. Trust covers transparency, legitimacy, platform credibility, and privacy-related concerns. Service covers customer support, usability, verification flow, technical errors, and application performance. Sentiment labels follow a three-class scheme (**Positive**, **Negative**, **Neutral**) and are only assigned when the target aspect is present; otherwise, the aspect is treated as absent (`null` in weak labeling and empty label with `aspect_present = 0` in the final gold subset).

