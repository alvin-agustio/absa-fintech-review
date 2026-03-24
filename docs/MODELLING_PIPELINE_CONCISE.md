# Modelling Pipeline, Singkat Tapi Lengkap

Dokumen ini menjelaskan bagaimana pipeline modelling di repo ini berjalan dari awal sampai akhir, dengan bahasa sederhana.

## Ringkasan Alur

1. Review Google Play diambil dari `Kredivo` dan `Akulaku`.
2. Review dibersihkan: dedup, hapus URL/emoji/artifact, lowercase, rapikan whitespace, buang review kosong atau terlalu pendek.
3. Corpus final dibentuk menjadi `v2`.
4. Weak label historis disejajarkan ke corpus final dengan `review_id` intersection.
5. Dari cohort review-level final, data diubah menjadi sample aspect-level untuk tiga aspek:
   - `risk`
   - `trust`
   - `service`
6. Model dilatih dengan dua jalur:
   - full fine-tuning IndoBERT
   - LoRA / PEFT IndoBERT
7. Setelah itu ada tahap uncertainty filtering untuk mencari sample yang kemungkinan noisy.
8. Sample clean dipakai lagi untuk retraining:
   - retrained full fine-tuning
   - retrained LoRA
9. Semua model dibandingkan di dua benchmark:
   - weak-label test set
   - gold subset manual
10. Model yang dipakai dashboard dipilih berdasarkan benchmark gold, bukan weak-label saja.

## Trace Data ke Modelling

- Raw reviews: `505,936`
- Final clean corpus `v2`: `270,329`
- Official review-level cohort: `48,763`
- Review dengan minimal satu label aspek: `42,313`
- Aspect-level training rows: `53,366`
- Noisy candidates dari uncertainty filtering: `1,117`
- Clean aspect-level rows setelah filtering: `52,249`

Maknanya sederhana:

- `48,763` adalah cohort resmi di level review
- `53,366` adalah sample yang benar-benar masuk modelling di level `(review, aspect)`
- `52,249` adalah versi clean subset untuk retraining

## Cara Sample Dibentuk

Satu review bisa menghasilkan sampai tiga sample.

Contoh:

- review: "bunga tinggi tapi CS cepat"
- sample 1: `[ASPECT=risk] bunga tinggi tapi CS cepat`
- sample 2: `[ASPECT=service] bunga tinggi tapi CS cepat`

Jadi model tidak membaca review mentah saja, tetapi membaca `review + tag aspek`.

## Jalur Model

### 1. Baseline

- IndoBERT full fine-tuning
- semua parameter model ikut di-update

### 2. LoRA

- IndoBERT + adapter LoRA
- hanya sebagian kecil parameter yang trainable
- dari artifact repo, trainable parameter LoRA sekitar `0.47%`

### 3. Retrained

- model baseline dilatih ulang di clean subset hasil uncertainty filtering

### 4. Retrained LoRA

- model LoRA dilatih ulang di clean subset

## Cara Menjelaskan 12 Variasi Model

Bagian ini yang paling penting untuk membuat eksperimen terasa rapi.

Sebenarnya ini **bukan 12 model yang berbeda arah**.
Ini adalah **4 keluarga model**, dan setiap keluarga diuji pada **3 jumlah epoch**.

Jadi bentuknya:

`4 families x 3 epochs = 12 runs`

### Empat keluarga model

1. `baseline`
   - full fine-tuning pada weak-label dataset utama
2. `lora`
   - LoRA pada weak-label dataset utama
3. `retrained`
   - full fine-tuning pada clean subset hasil uncertainty filtering
4. `retrained_lora`
   - LoRA pada clean subset hasil uncertainty filtering

### Tiga setting epoch

- `epoch 3`
- `epoch 5`
- `epoch 8`

### Matriks sederhana

```text
                     Epoch 3         Epoch 5         Epoch 8
Baseline             baseline_3      baseline_5      baseline_8
LoRA                 lora_3          lora_5          lora_8
Retrained            retrained_3     retrained_5     retrained_8
Retrained LoRA       retr_lora_3     retr_lora_5     retr_lora_8
```

Jadi cara paling mudah menjelaskannya adalah:

- sumbu pertama: **jenis training**
- sumbu kedua: **lama training**

Bukan 12 eksperimen acak, tetapi 12 kombinasi dari dua keputusan eksperimen.

## Narasi Penjelasan yang Paling Aman

Kalau ingin menjelaskan di paper, slide, atau sidang, narasi paling sederhana adalah:

> Eksperimen ini menguji empat keluarga model: baseline full fine-tuning, LoRA, retrained full fine-tuning, dan retrained LoRA. Setiap keluarga diuji pada tiga jumlah epoch, yaitu 3, 5, dan 8. Dengan demikian, total ada 12 run. Struktur ini dipakai untuk menjawab dua pertanyaan: apakah clean-subset retraining membantu, dan apakah LoRA bisa menyaingi full fine-tuning.

Kalau ingin versi yang lebih singkat:

> Dua faktor utama yang diuji adalah **training regime** dan **jumlah epoch**. Kombinasi keduanya menghasilkan 12 run.

## Visual yang Paling Tepat

Kalau ingin visual yang tidak membingungkan, jangan mulai dari leaderboard 12 model sekaligus.
Mulailah dari struktur eksperimennya dulu.

### Visual 1. Matrix overview

Gunakan tabel 4 x 3 seperti ini:

```text
                3 epoch      5 epoch      8 epoch
Baseline          yes          yes          yes
LoRA              yes          yes          yes
Retrained         yes          yes          yes
Retrained LoRA    yes          yes          yes
```

Fungsi visual ini:

- membuat pembaca langsung paham struktur 12 run
- mengurangi kesan bahwa ada "terlalu banyak model acak"

### Visual 2. Grouped bar chart per family

Visual terbaik untuk hasil akhir biasanya:

- sumbu X: `Baseline`, `LoRA`, `Retrained`, `Retrained LoRA`
- dalam tiap grup ada 3 bar: `epoch 3`, `epoch 5`, `epoch 8`
- sumbu Y: `F1 Macro`

Kenapa ini bagus:

- pembaca bisa lihat efek epoch di dalam keluarga yang sama
- pembaca bisa lihat apakah retraining membantu
- pembaca bisa lihat apakah LoRA lebih efisien atau lebih kuat

### Visual 3. Pisahkan benchmark

Jangan campur weak-label dan gold dalam satu chart utama.

Lebih aman pakai dua chart:

1. `Weak-label F1 Macro`
2. `Gold-subset F1 Macro`

Alasannya:

- weak-label dan gold menjawab pertanyaan yang berbeda
- kalau digabung, pembaca mudah salah baca

## Solusi Penjelasan yang Saya Rekomendasikan

Kalau Anda ingin presentasi yang paling rapi, gunakan urutan ini:

1. Tunjukkan dulu matrix 4 x 3
   - supaya pembaca paham struktur eksperimennya
2. Tunjukkan grouped bar chart weak-label
   - untuk menjawab siapa winner di weak benchmark
3. Tunjukkan grouped bar chart gold
   - untuk menjawab siapa winner di human validation
4. Tutup dengan satu kalimat utama
   - weak winner tidak sama dengan gold winner

## Inti Pesan yang Harus Ditangkap Pembaca

Pembaca tidak perlu menghafal 12 nama model.

Yang mereka perlu pahami hanya tiga hal:

1. ada **4 keluarga model**
2. masing-masing diuji di **3 epoch**
3. winner di weak-label dan winner di gold-subset **tidak sama**

## Kenapa Ada Dua Benchmark

### Weak-label benchmark

Ini mengukur seberapa cocok model terhadap label weak hasil pipeline.

Kelebihan:

- besar
- stabil untuk eksperimen awal

Kekurangan:

- belum tentu paling dekat dengan judgment manusia

### Gold-subset benchmark

Ini mengukur performa pada subset manual berlabel manusia.

Kelebihan:

- lebih dekat ke kualitas nyata

Kekurangan:

- ukurannya kecil

Insight paling penting dari repo ini:

> model terbaik di weak-label tidak sama dengan model terbaik di gold subset.

## Comparison Antar Model

### A. Weak-label test set

Metrik utama: `F1 Macro`

```text
retrained_lora_epoch8  0.8787  ##################
retrained_epoch8       0.8724  #################
retrained_lora_epoch5  0.8715  #################
retrained_epoch5       0.8644  #################
retrained_epoch3       0.8620  #################
retrained_lora_epoch3  0.8459  ################
baseline_epoch5        0.7933  ############
lora_epoch8            0.7895  ############
baseline_epoch8        0.7866  ############
lora_epoch5            0.7849  ############
baseline_epoch3        0.7805  ############
lora_epoch3            0.7799  ############
```

Weak-label winner:

- `retrained_lora_epoch8`

### B. Gold subset manual

Metrik utama: `Sentiment F1 Macro (present rows only)`

```text
lora_epoch8            0.8174  ################
retrained_epoch8       0.8097  ################
baseline_epoch8        0.7946  ###############
baseline_epoch5        0.7901  ###############
lora_epoch5            0.7477  ##############
retrained_epoch3       0.7436  ##############
retrained_epoch5       0.7310  ##############
retrained_lora_epoch8  0.7076  ##############
baseline_epoch3        0.7060  ##############
retrained_lora_epoch5  0.6657  #############
lora_epoch3            0.6144  ############
retrained_lora_epoch3  0.6098  ############
```

Gold winner:

- `lora_epoch8`

## Interpretasi Singkat

- Jika hanya melihat weak-label, model terbaik adalah `retrained_lora_epoch8`.
- Jika melihat validasi manusia, model terbaik adalah `lora_epoch8`.
- Karena dashboard dipakai untuk insight yang dibaca manusia, repo ini lebih mengutamakan hasil gold subset saat memilih model default.

## Source of Truth di Repo

Data dan artefak utama yang dipakai untuk pipeline ini:

- `data/processed/dataset_absa_50k_v2_intersection.csv`
- `droplet/skripsi_eval_core/data/processed/evaluation/epoch_comparison_summary.csv`
- `data/processed/diamond/evaluation_all_models/gold_evaluation_overview.csv`
- `droplet/skripsi_eval_core/data/processed/noise/noise_summary.json`

Script training utama:

- `src/training/train_baseline.py`
- `src/training/train_lora.py`
- `src/training/retrain_filtered.py`
- `src/training/train_lora_filtered.py`

## Kesimpulan

Pipeline modelling repo ini bisa dibaca sebagai:

`raw reviews -> cleaned final corpus -> weak-labeled cohort -> aspect-level training data -> baseline/LoRA training -> uncertainty filtering -> retraining -> gold validation -> pilih model final`

Kalau disederhanakan lagi:

- baseline dan LoRA dilatih di dataset weak-label utama
- data noisy disaring
- model dilatih ulang di clean subset
- keputusan final model tidak diambil dari weak-label saja, tetapi dicek lagi dengan gold subset manual
