# Dataset Tables for Paper

Dokumen ini merangkum tabel dan figure yang paling penting untuk menjelaskan dataset secara cepat di paper. Semua angka di bawah ini disusun berdasarkan artefak yang ada di repo saat ini.

Sumber utama yang dipakai:

- `data/raw/reviews_raw.csv`
- `data/processed/reviews_clean_v2.csv`
- `data/processed/dataset_absa_v2.csv`
- `data/processed/dataset_absa_50k_v2_intersection.csv`
- `droplet/skripsi_eval_core/data/processed/noise/noise_summary.json`

---

## Table 1. Dataset Size Summary

Tabel ini merangkum alur jumlah data dari awal hingga akhir pipeline.

| Stage | Number of Instances |
|---|---:|
| Raw reviews | 505,936 |
| Cleaned reviews (v2) | 270,329 |
| Official training dataset (review-level) | 48,763 |
| Aspect-level instances | 53,366 |
| Flagged noisy candidates | 1,117 |
| Filtered subset | 52,249 |

Catatan singkat:

- `Raw reviews` berasal dari hasil scrape Google Play.
- `Cleaned reviews (v2)` adalah corpus setelah cleaning dan normalisasi v2.
- `Official training dataset` adalah cohort review-level resmi yang dipakai untuk eksperimen utama.
- `Aspect-level instances` adalah hasil transformasi review-level menjadi pasangan `(review, aspect)` untuk training.
- `Flagged noisy candidates` dan `Filtered subset` berasal dari tahap uncertainty-aware filtering.

---

## Table 2. Aspect and Sentiment Distribution

Tabel ini menunjukkan distribusi data pada level aspect-level dataset yang benar-benar dipakai dalam modeling.

| Aspect | Negative | Positive | Neutral |
|---|---:|---:|---:|
| Risk | 10,454 | 3,158 | 748 |
| Service | 13,676 | 19,097 | 450 |
| Trust | 3,392 | 2,225 | 166 |
| **Total** | **27,522** | **24,480** | **1,364** |

Catatan singkat:

- `Service` adalah aspek dengan jumlah instance terbesar.
- `Neutral` sangat sedikit dibanding `Negative` dan `Positive`.
- Tabel ini memperlihatkan imbalance utama dataset secara eksplisit.

---

## Table 3. Example Rows and Dataset Schema

Tabel ini memperlihatkan contoh nyata bentuk row pada dataset training resmi `dataset_absa_50k_v2_intersection.csv`.

| review_id | app_name | review_date | review_text | risk_sentiment | trust_sentiment | service_sentiment |
|---:|---|---|---|---|---|---|
| 126840 | Akulaku | 2024-03-05 | `gx di acc` | - | - | Negative |
| 250334 | Kredivo | 2024-05-07 | `baik dc nya` | Positive | - | - |
| 59685 | Kredivo | 2025-05-26 | `aman n cepat` | - | Positive | Positive |
| 100122 | Akulaku | 2025-09-18 | `bunganya gede coiii` | Negative | - | - |

Catatan singkat:

- Tanda `-` berarti aspek tersebut tidak mendapat label pada row review-level itu.
- Satu review bisa memiliki lebih dari satu label aspek, seperti pada `review_id = 59685`.
- Karena itu, row review-level kemudian diubah lagi menjadi row aspect-level sebelum training model.

---

## Table 4. Combined Aspect-Level Distribution Layout

Tabel ini dibuat dengan gaya yang lebih mirip contoh visual referensi: header utama adalah kelas sentimen, dan subkolomnya adalah aspek.

**Table 4. Final aspect-level data distribution based on sentiment and aspect class**

<table>
  <thead>
    <tr>
      <th>Sentiment Class</th>
      <th colspan="3">Negative</th>
      <th colspan="3">Positive</th>
      <th colspan="3">Neutral</th>
    </tr>
    <tr>
      <th>Aspect Class</th>
      <th>Risk</th>
      <th>Service</th>
      <th>Trust</th>
      <th>Risk</th>
      <th>Service</th>
      <th>Trust</th>
      <th>Risk</th>
      <th>Service</th>
      <th>Trust</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Total Data</td>
      <td>10,454</td>
      <td>13,676</td>
      <td>3,392</td>
      <td>3,158</td>
      <td>19,097</td>
      <td>2,225</td>
      <td>748</td>
      <td>450</td>
      <td>166</td>
    </tr>
  </tbody>
</table>

---

## Figure 1. Sentiment and Aspect Distribution Visualization

Figure ini adalah versi visual sederhana dari distribusi utama dataset. Bagian kiri menunjukkan distribusi sentiment, dan bagian kanan menunjukkan distribusi aspect.

![Figure 1. Sentiment and aspect total data distribution and percentage based on classes.](/C:/Users/alvin/Downloads/skripsi/docs/dataset_distribution_figure.png)

Interpretasi singkat:

- `Negative` dan `Positive` mendominasi dataset.
- `Neutral` jauh lebih kecil, sehingga wajar menjadi kelas yang paling sulit diprediksi saat evaluasi.
- `Service` adalah aspek dengan jumlah instance terbesar, diikuti `Risk`, lalu `Trust`.
