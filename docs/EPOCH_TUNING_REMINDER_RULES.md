# Epoch Tuning Reminder Rules

Status: active reminder for the next experiment cycle

Tujuan dokumen ini adalah menjadi pengingat singkat dan operasional untuk prosedur epoch tuning sebelum retraining `baseline`, `LoRA`, `DoRA`, `QLoRA`, dan `AdaLoRA`.

Dokumen ini dipakai sebagai aturan kerja praktis. Jika ada perbedaan dengan eksperimen lama, maka aturan di dokumen ini yang diikuti untuk siklus tuning berikutnya.

## Core Goal

Pemilihan epoch tidak boleh dilakukan secara arbitrer seperti hanya mencoba `3`, `5`, dan `8` tanpa landasan. Epoch harus dipilih dari hasil validasi, sambil menjaga dua hal:

- menghindari overfitting
- mengukur efisiensi training, terutama untuk membandingkan full fine-tuning vs PEFT

## Fixed Rules

Aturan yang dikunci untuk tuning berikutnya:

- `max epoch = 15`
- evaluasi dilakukan pada setiap epoch
- checkpoint terbaik disimpan berdasarkan metrik validasi
- tidak menggunakan early stopping
- split `train`, `validation`, dan `test` harus tetap
- test set tidak boleh dipakai untuk memilih epoch

## Primary Selection Logic

Checkpoint terbaik dipilih berdasarkan urutan prioritas berikut:

1. `validation Macro-F1`
2. `validation Accuracy`
3. `validation Precision`
4. `validation Recall`
5. efisiensi waktu training
6. jika masih sangat mirip, pilih epoch yang lebih kecil

Catatan:

- `Macro-F1` tetap menjadi metrik utama karena task sentimen tidak selalu seimbang.
- `Accuracy`, `Precision`, dan `Recall` dipakai sebagai metrik pendukung agar pembacaan performa lebih lengkap.
- waktu training ikut dihitung karena salah satu tujuan eksperimen adalah melihat efisiensi PEFT.

## Time and Efficiency Logging

Karena salah satu target eksperimen adalah membandingkan efisiensi PEFT, maka setiap run wajib mencatat waktu training dengan jelas.

Minimal yang harus dicatat:

- waktu mulai training
- waktu selesai training
- durasi total training
- timestamp per epoch
- durasi per epoch
- epoch terbaik

Interpretasi yang diharapkan:

- berapa lama model mencapai epoch terbaik
- berapa lama total 15 epoch bila training dijalankan penuh
- seberapa hemat PEFT dibandingkan baseline full fine-tuning

## Metrics To Log Every Epoch

Setiap epoch minimal harus menghasilkan catatan berikut:

- `train loss`
- `validation loss`
- `validation Macro-F1`
- `validation Accuracy`
- `validation Precision`
- `validation Recall`
- `epoch duration`

Jika memungkinkan, simpan juga:

- `cumulative training time`
- nama checkpoint per epoch

## Overfitting Reading Guide

Walaupun tidak memakai early stopping, risiko overfitting tetap harus diamati dari kurva train dan validation.

Indikasi overfitting yang perlu dicermati:

- `train loss` terus turun, tetapi `validation loss` mulai naik
- `train loss` terus turun, tetapi `validation Macro-F1` stagnan atau turun
- metric validasi terbaik muncul lebih awal, lalu menurun pada epoch-epoch berikutnya

Artinya:

- model mungkin semakin hafal data train
- tetapi tidak semakin baik saat diuji pada data validation

## Fair Comparison Rules Across Models

Agar perbandingan `baseline`, `LoRA`, `DoRA`, `QLoRA`, dan `AdaLoRA` tetap adil, semua kandidat harus mengikuti prosedur yang sama:

- split data sama
- `max epoch` sama
- frekuensi evaluasi sama, yaitu setiap epoch
- aturan simpan checkpoint sama
- metrik pemilihan sama
- format input sama

Catatan penting:

- epoch terbaik antar model tidak harus sama
- yang harus sama adalah prosedurnya, bukan hasil akhirnya

## Train-Validation-Test Discipline

Pembagian fungsi dataset harus dijaga dengan ketat:

- `train` hanya untuk melatih model
- `validation` untuk memilih checkpoint dan membaca kurva belajar
- `test` hanya untuk evaluasi akhir

Larangan:

- jangan gunakan test set untuk memilih epoch
- jangan ubah split data di tengah siklus tuning

## Tie-Break Rule

Jika dua checkpoint terlihat sangat dekat, gunakan aturan berikut:

1. pilih `Macro-F1` validasi yang lebih tinggi
2. jika sama, pilih `Accuracy` lebih tinggi
3. jika sama, pilih `Precision` lebih tinggi
4. jika sama, pilih `Recall` lebih tinggi
5. jika sama, pilih waktu training yang lebih hemat
6. jika semua masih sama, pilih epoch yang lebih kecil

## Scope Discipline

Tahap ini fokus pada tuning epoch dan pembacaan efisiensi. Karena itu:

- jangan ubah taxonomy
- jangan ubah split data
- jangan ubah format input
- jangan tuning terlalu banyak hyperparameter lain sekaligus jika belum perlu

Kalau ada eksperimen lain seperti perubahan learning rate atau batch size, pisahkan sebagai eksperimen berbeda agar interpretasinya tetap bersih.

## Output That Must Exist

Sebelum hasil dianggap final, setiap model harus punya artefak berikut:

- log metric per epoch
- log waktu per epoch
- checkpoint terbaik
- ringkasan epoch terbaik
- hasil evaluasi test
- hasil evaluasi gold subset

## Practical Thesis Statement

Kalimat sederhana yang bisa dipakai saat menulis metodologi:

> Batas maksimum pelatihan ditetapkan sebesar 15 epoch sebagai ruang eksplorasi yang seragam untuk seluruh model. Evaluasi dilakukan pada setiap epoch, dan checkpoint terbaik dipilih berdasarkan nilai validation Macro-F1 dengan mempertimbangkan Accuracy, Precision, Recall, serta efisiensi waktu training sebagai metrik pendukung. Prosedur ini digunakan untuk menghindari pemilihan epoch yang arbitrer sekaligus memungkinkan perbandingan efisiensi antara full fine-tuning dan PEFT.

## Final Reminder

Inti aturan ini adalah:

- epoch final tidak dipilih dari intuisi
- epoch final tidak dipilih dari test set
- model terbaik tidak cukup dilihat dari skor saja, tetapi juga dari efisiensi training
- tujuan akhirnya bukan sekadar skor tertinggi, tetapi model yang kuat, stabil, dan efisien
