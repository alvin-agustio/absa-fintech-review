# Project Status (Updated)

## 1. Current Truth Sources
- Active training dataset: `data/processed/dataset_absa_50k_v2_intersection.csv`
- Active preprocessing output: `data/processed/reviews_clean_v2.csv`
- Active v2 full dataset: `data/processed/dataset_absa_v2.csv`
- Gold manual template (single annotator): `data/processed/diamond/template_anotator_tunggal_balanced_300_aspect_rows.csv`

## 2. Provenance and Archive
- Historical silver v1 dataset is preserved in archive for reproducibility:
  - `data/processed/archive/2026-03-13_focus_v2/dataset_absa.csv`
- Historical model outputs are preserved in:
  - `models/archive/2026-03-13_focus_v2/`

## 3. Model and Config Consistency
- Active backbone in code: `indobenchmark/indobert-base-p1` (see `config.py`)
- Training defaults now point to active v2 training dataset.

## 4. Pipeline Stage (Actual)
- Data preprocessing v2: completed
- 50k cohort intersection to v2: completed
- Training baseline/LoRA on v2 dataset: ready to run (folders reset for fresh runs)
- Gold annotation subset prep: completed and synced to v2 intersection IDs

## 5. Evaluation Behavior
- `evaluate.py` first reads active model dirs:
  - `models/baseline`, `models/lora`, `models/retrained`
- If active dirs are empty, it now falls back to the latest archive snapshot automatically.

## 6. Notes for Thesis Writing
- Do not claim labels were generated directly from v2.
- Correct statement: labels were generated in v1, then reconciled to v2 via review_id intersection.
- Gold subset is single-annotator gold (not full diamond standard).

_Last Updated: 2026-03-13_
