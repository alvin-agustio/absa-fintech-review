# Project Context (Ringkas)

## Tujuan
Membangun pipeline ABSA untuk ulasan fintech (`Kredivo`, `Akulaku`) pada 3 aspek:
- `risk`
- `trust`
- `service`

Label sentimen:
- `Negative`
- `Neutral`
- `Positive`

---

## Alur Proses Yang Sudah Dijalankan

1. Scraping dan preprocessing
- Raw data dikumpulkan dari Play Store.
- Versi aktif clean saat ini: `data/processed/reviews_clean_v2.csv`.

2. Labeling silver (LLM)
- Labeling massal dilakukan pada v1.
- Dataset aktif saat ini dibentuk ulang pada v2 melalui irisan `review_id`:
  - `data/processed/dataset_absa_v2.csv`
  - `data/processed/dataset_absa_50k_v2_intersection.csv` (dataset resmi training)

3. Training eksperimen
- Baseline: `train_baseline.py`.
- LoRA: `train_lora.py`.
- Default input training sekarang: `data/processed/dataset_absa_50k_v2_intersection.csv`.
- Hasil eksperimen lama dipindah ke archive snapshot; folder `models/baseline`, `models/lora`, `models/retrained` dipakai untuk run aktif berikutnya.

4. Evaluasi
- Perbandingan model via `evaluate.py`.
- Metrik utama: `test_f1_macro`, `test_accuracy`, `test_f1_weighted`.

5. Persiapan gold subset manual
- Dibuat template anotasi tunggal dari data final.
- File kerja anotasi saat ini:
  - `data/processed/diamond/template_anotator_tunggal.csv`
  - `data/processed/diamond/pedoman_anotasi_diamond.md`

---

## Status Eksperimen Saat Ini

- Pipeline v2 selesai dan aktif.
- Source-of-truth training saat ini: `dataset_absa_50k_v2_intersection.csv`.
- Source-of-truth validasi manual saat ini: `template_anotator_tunggal_balanced_300_aspect_rows.csv`.
- Eksperimen lama tersimpan di archive untuk provenance.

---

## Struktur Folder (Rapi)

- `data/processed/` -> dataset final, manifest, evaluasi, file anotasi manual.
- `models/` -> output model baseline, lora, retrained, dan arsip model lama.
- `docs/` -> dokumentasi proyek lama (tasklist/guidelines/status).
- `scripts/` -> runner script eksperimen.
- `doc.md` -> ringkasan konteks proses terkini (file ini).

---

## Catatan Metodologi

- `dataset_absa.csv` (v1) adalah sumber label silver historis dan sekarang berada di archive.
- Dataset resmi training aktif adalah `dataset_absa_50k_v2_intersection.csv`.
- Subset manual Anda adalah **gold subset** (single annotator), bukan diamond multi-annotator.
- Gunakan gold subset untuk validasi kualitas model terhadap label manusia.

---

## Next Step Disarankan

1. Selesaikan anotasi manual pada `template_anotator_tunggal.csv`.
2. Simpan hasil final sebagai `gold_final.csv`.
3. Evaluasi model terbaik (mis. LoRA epoch 5) terhadap gold subset.
4. Laporkan gap performa antara evaluasi silver vs gold di laporan skripsi.
