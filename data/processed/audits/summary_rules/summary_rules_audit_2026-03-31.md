# Summary Rules Audit

- Generated at: `2026-03-31T20:00:57`
- Source: `latest cache pair: ff692c341dca_reviews.csv + ff692c341dca_predictions.csv`
- Status: `ready`
- Unique reviews: `6086`
- Apps: `Akulaku, Kredivo`
- Aspects: `risk, trust, service`
- Quality: `medium`

## Warnings
- Tidak ada kolom issue, jadi bagian isu disederhanakan.

## Guardrails
- [PASS] both_apps_present: Kredivo dan Akulaku sama-sama muncul di summary.
- [PASS] app_order_is_kredivo_first: Urutan app sudah Kredivo lalu Akulaku.
- [PASS] app_cards_name_anchored: Setiap kartu app menyebut nama aplikasinya sendiri.
- [PASS] overall_mentions_anchor_aspects: Overall summary mengikat aspek kuat (Trust) dan aspek lemah (Risk).
- [PASS] signal_mentions_worst_aspect: Blok sinyal menyorot aspek Risk sebagai prioritas.
- [PASS] meaning_mentions_app_contrast: Makna akhir sudah membedakan Kredivo dan Akulaku.
- [PASS] app_contrast_visible: Kontras app terlihat lewat gap sentimen (16.3 pts positif, 17.4 pts negatif), similarity teks 0.78.
- [INFO] issue_column_available: Kolom issue tidak tersedia; summary memakai label aspek/hint sebagai fallback.
- [PASS] data_sufficiency: Ukuran data cukup untuk summary ringkas dan perbandingan app.

## Summary Preview

**Overall**: Secara umum, pengalaman pengguna cenderung positif dan terasa membantu untuk kebutuhan harian. Sisi yang paling sering dipuji ada pada aspek Trust.  Ini menunjukkan kejelasan proses dan rasa aman masih menjadi nilai yang cukup terasa. Tekanan terbesar ada pada aspek Risk. Dari dua aplikasi, Kredivo terlihat lebih stabil daripada Akulaku. Keluhan yang paling sering muncul terkait limit, approval, pencairan, dan biaya. Ini menunjukkan proses pembiayaan masih belum selalu terasa mulus, jelas, atau menenangkan bagi pengguna.

**Signal**: Perhatian utama ada pada aspek Risk. Tekanan negatif paling kuat muncul di area limit, approval, pencairan, dan biaya. Keluhan yang paling sering muncul terkait limit, approval, pencairan, dan biaya. Ini menunjukkan proses pembiayaan masih belum selalu terasa mulus, jelas, atau menenangkan bagi pengguna. Pergerakan sentimen relatif stabil dalam beberapa hari terakhir.

**Meaning**: Artinya, aplikasi ini sudah punya fondasi pengalaman yang cukup baik. Hal yang paling membantu ada pada aspek Trust, sementara titik yang paling sering memicu keluhan ada pada aspek Risk. Secara praktis, Kredivo terlihat lebih tenang, sedangkan Akulaku masih butuh perhatian lebih. Keluhan yang paling sering muncul terkait limit, approval, pencairan, dan biaya. Ini menunjukkan proses pembiayaan masih belum selalu terasa mulus, jelas, atau menenangkan bagi pengguna. Kalau pola ini terus berulang, kualitas pengalaman pakai dan rasa percaya terhadap layanan bisa ikut melemah.

### Per-App Snapshot

| app_name | tone | dominant_sentiment | positive_share | neutral_share | negative_share | best_aspect | worst_aspect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kredivo | positive | Positive | 63.4 | 7.8 | 28.8 | trust | risk |
| Akulaku | neutral | Positive | 47.1 | 6.6 | 46.2 | trust | risk |

## Per-App Text
### Kredivo
Di Kredivo, pengalaman pengguna cenderung positif dan cukup membantu untuk kebutuhan harian. Kekuatan paling terasa ada pada aspek Trust.  Ini menunjukkan kejelasan proses dan rasa aman masih menjadi nilai yang cukup terasa. Perhatian utama ada pada aspek Risk. Keluhan yang paling sering muncul terkait limit, approval, pencairan, dan biaya. Ini menunjukkan proses pembiayaan masih belum selalu terasa mulus, jelas, atau menenangkan bagi pengguna. Pergerakan sentimen relatif stabil dalam beberapa hari terakhir.

### Akulaku
Di Akulaku, pengalaman pengguna masih campuran, tetapi sisi positif sedikit unggul dan cukup membantu untuk kebutuhan harian. Kekuatan paling terasa ada pada aspek Trust.  Ini menunjukkan kejelasan proses dan rasa aman masih menjadi nilai yang cukup terasa. Perhatian utama ada pada aspek Risk. Keluhan yang paling sering muncul terkait limit, approval, pencairan, dan biaya. Ini menunjukkan proses pembiayaan masih belum selalu terasa mulus, jelas, atau menenangkan bagi pengguna. Tekanan negatif cenderung turun dalam beberapa hari terakhir.
