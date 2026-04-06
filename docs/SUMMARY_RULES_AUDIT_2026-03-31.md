# Summary Rules Audit

Audit ini memotret output rule-based summary dari cache dashboard terbaru dan menyimpannya sebagai artefak review di:

- `data/processed/audits/summary_rules/summary_rules_audit_2026-03-31.md`
- `data/processed/audits/summary_rules/summary_rules_audit_2026-03-31.json`

## Verdict

- Status summary: `ready`
- Coverage: `6,086` review unik
- Apps yang terdeteksi: `Kredivo` dan `Akulaku`
- Aspek yang aktif: `Risk`, `Trust`, `Service`

## Temuan Utama

- Summary sudah membedakan Kredivo dan Akulaku secara jelas.
- `Risk` tetap menjadi titik perhatian terbesar.
- `Trust` muncul sebagai sisi yang paling kuat.
- Karena kolom `issue` tidak tersedia di cache terbaru, kalimat isu masih memakai fallback berbasis aspek dan hint taxonomy.
- Near-tie pada level app dibaca lebih hati-hati sebagai campuran, bukan dominan penuh.

## Cara Menjalankan Ulang

```bash
python scripts/audit_summary_rules.py
```

## Tes Terkait

- `tests/test_summary_rules.py`
- `tests/test_summary_audit.py`
