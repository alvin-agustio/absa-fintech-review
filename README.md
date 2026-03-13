# ABSA Fintech Review Analysis

Aspect-Based Sentiment Analysis for Google Play Store reviews of Indonesian fintech lending apps, focused on three aspects: `risk`, `trust`, and `service`.

This repository contains an end-to-end experimental pipeline covering review scraping, preprocessing, silver-label construction, IndoBERT baseline training, LoRA fine-tuning, evaluation, and a Streamlit demo app.

## Highlights

- Domain: Indonesian fintech app reviews (`Kredivo` and `Akulaku`)
- Task: 3-aspect ABSA with 3 sentiment classes (`Negative`, `Neutral`, `Positive`)
- Backbone: `indobenchmark/indobert-base-p1`
- Training tracks: full fine-tuning baseline and LoRA
- App demo: interactive Streamlit dashboard for live review analysis
- Methodology note: active training data uses the v2-cleaned corpus reconciled with historical silver labels via `review_id` intersection

## Repository Scope

The public repository is intended as a portfolio and reproducibility-oriented codebase.

Large artifacts such as trained models, raw scraped data, and full processed datasets are excluded from version control to keep the repository lightweight and clone-friendly.

## Project Structure

```text
.
|- app.py
|- config.py
|- preprocess.py
|- labeling.py
|- train_baseline.py
|- train_lora.py
|- evaluate.py
|- inference.py
|- detect_label_noise.py
|- predict_mc_dropout.py
|- retrain_filtered.py
|- scripts/
|- docs/
|- data/
```

## Workflow

1. Scrape and clean Google Play reviews.
2. Build the active v2 normalized corpus.
3. Reconcile historical silver labels into v2 via `review_id` intersection.
4. Train IndoBERT baseline and LoRA models.
5. Evaluate on the silver split.
6. Validate final selected model on a manually annotated gold subset.

## Key Files

- `train_baseline.py`: baseline full fine-tuning
- `train_lora.py`: LoRA fine-tuning
- `evaluate.py`: experiment comparison and evaluation summary
- `app.py`: Streamlit dashboard
- `scripts/run_baseline_epochs.ps1`: baseline epoch sweep
- `scripts/run_lora_epochs.ps1`: LoRA epoch sweep
- `scripts/run_training_experiments.ps1`: combined experiment runner
- `docs/PROJECT_STATUS.md`: current project status
- `doc.md`: concise active-context document

## Quick Start

### 1. Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run training

Baseline sweep:

```powershell
.\scripts\run_baseline_epochs.ps1
```

LoRA sweep:

```powershell
.\scripts\run_lora_epochs.ps1
```

Run both:

```powershell
.\scripts\run_training_experiments.ps1
```

### 3. Run evaluation

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

### 4. Run dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Data Note

The active experiment configuration in code points to the v2 training dataset:

- `data/processed/dataset_absa_50k_v2_intersection.csv`

That file is part of the local research workspace but should generally not be committed to the public repository if you want the repo to stay lightweight.

## Methodology Note

- Historical silver labels were generated on v1.
- The active v2 experiment dataset is built by intersecting v1 silver labels with the v2-cleaned corpus.
- Manual annotation files represent a single-annotator gold subset, not a full diamond annotation setup.

## Portfolio Positioning

This project is suitable to present as a portfolio item for:

- NLP experimentation
- Indonesian-language text classification
- ABSA pipeline design
- LLM-assisted data labeling workflow
- Parameter-efficient fine-tuning with LoRA
- Streamlit-based ML demo deployment

## Next Improvements

- Add a small public sample dataset for reproducible demo runs
- Add saved evaluation tables or figures to `docs/`
- Add a license file before publishing publicly
- Add GitHub Actions for basic linting and smoke tests