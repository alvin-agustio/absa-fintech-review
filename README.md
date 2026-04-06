# Fintech Review ABSA

![Status](https://img.shields.io/badge/status-on%20progress-ffb703)
![Task](https://img.shields.io/badge/task-ABSA-219ebc)
![Language](https://img.shields.io/badge/language-Indonesian-2a9d8f)
![Model](https://img.shields.io/badge/model-IndoBERT%20base%20p1-264653)
![Training](https://img.shields.io/badge/training-Baseline%20%2B%20LoRA-e76f51)

An end-to-end Aspect-Based Sentiment Analysis pipeline for Google Play Store reviews of Indonesian fintech lending apps.

This project focuses on three business-relevant aspects:

- `risk`
- `trust`
- `service`

with three sentiment labels:

- `Negative`
- `Neutral`
- `Positive`

The repository is being developed as both:

- an active thesis/research workspace
- a public portfolio project for NLP, model experimentation, and applied ML engineering

## Why This Project

Most sentiment projects stop at generic positive-vs-negative classification. This one pushes further into aspect-level analysis on Indonesian fintech reviews, where user language is noisy, colloquial, and often mixes product trust, service quality, and perceived risk in the same review.

That makes the project useful for demonstrating:

- practical ABSA pipeline design
- weak-label to cleaner-dataset reconciliation
- IndoBERT fine-tuning and LoRA comparison
- research-oriented experimentation with portfolio-grade packaging

## Current Status

This repository is still **on progress**.

What is already in place:

- v2 preprocessing pipeline
- active v2 training dataset configuration
- baseline and LoRA training scripts
- evaluation pipeline
- Streamlit dashboard
- manual gold subset preparation

What is still ongoing:

- fresh training runs on the active v2 setup
- final comparison on manually annotated gold data
- stronger public demo assets such as screenshots and sample outputs

For the canonical experiment procedure after taxonomy freeze, see `docs/MODEL_EPOCH_AND_EXPERIMENT_PROTOCOL_2026-03-31.md`. If you already have an `epoch_comparison_summary.csv`, the helper `scripts/recommend_epoch_from_epoch_sweep.py` can turn it into a readable epoch recommendation.

## Pipeline Overview

```mermaid
flowchart LR
	A[Google Play Reviews] --> B[Scraping]
	B --> C[Preprocessing v2]
	C --> D[Normalized Review Corpus]
	D --> E[Historical Silver Labels from v1]
	E --> F[review_id Intersection]
	F --> G[Active Training Dataset v2]
	G --> H1[Baseline Fine-Tuning]
	G --> H2[LoRA Fine-Tuning]
	H1 --> I[Evaluation on Silver Split]
	H2 --> I
	I --> J[Best Model Selection]
	J --> K[Manual Gold Annotation]
	K --> L[Final Validation Against Human Labels]
```

## Project Snapshot

| Area | Current Choice |
| --- | --- |
| Domain | Indonesian fintech app reviews |
| Apps | Kredivo, Akulaku |
| Task | 3-aspect ABSA |
| Labels | Negative, Neutral, Positive |
| Backbone | `indobenchmark/indobert-base-p1` |
| Training Tracks | Baseline full fine-tuning, LoRA |
| Active Dataset | `data/processed/dataset_absa_50k_v2_intersection.csv` |
| Final Validation Direction | Manual single-annotator gold subset |

## Experiment Design

```mermaid
flowchart TD
	A[reviews_clean_v2.csv] --> B[dataset_absa_v2.csv]
	B --> C[dataset_absa_50k_v2_intersection.csv]
	C --> D[train_baseline.py]
	C --> E[train_lora.py]
	D --> F[evaluate.py]
	E --> F
	F --> G[Compare metrics and errors]
	G --> H[Select best checkpoint]
	H --> I[Validate on gold subset]
```

## Repository Layout

```text
.
├── app.py                          # Streamlit dashboard (entry point)
├── config.py                       # central paths and model configuration
├── requirements.txt
│
├── src/                            # all pipeline Python source modules
│   ├── inference.py                # ABSAPredictor class used by app.py
│   ├── data/
│   │   ├── preprocess.py           # review text cleaning and normalization
│   │   ├── scrape_reviews.py       # Google Play scraper
│   │   ├── resume_scrape.py        # incremental scrape resume helper
│   │   └── labeling.py             # LLM-based silver labeling (Groq)
│   ├── training/
│   │   ├── train_baseline.py       # full fine-tuning pipeline
│   │   ├── train_lora.py           # LoRA fine-tuning pipeline
│   │   ├── train_lora_filtered.py  # LoRA on noise-filtered data
│   │   └── retrain_filtered.py     # retraining on clean subset
│   └── evaluation/
│       ├── evaluate.py             # experiment comparison and metrics
│       ├── detect_label_noise.py   # weak-label noise detection
│       └── predict_mc_dropout.py   # MC Dropout uncertainty estimation
│
├── scripts/                        # execution runners and data pipeline scripts
│   ├── run_baseline_epochs.ps1     # baseline epoch sweep
│   ├── run_lora_epochs.ps1         # LoRA epoch sweep
│   ├── run_training_experiments.ps1
│   ├── build_v2_intersection.py    # build active v2 training dataset
│   ├── audit_normalization_v2.py   # audit slang normalization coverage
│   └── setup_digitalocean_gpu.sh
│
├── docs/                           # project documentation
│   ├── PROJECT_STATUS.md
│   ├── CONTEXT.md                  # active project context summary
│   ├── DIAMOND_STANDARD_GUIDELINES.md
│   ├── DIGITALOCEAN_GPU_SETUP.md
│   └── EXECUTION_TASKLIST_UNTIL_EVALUATE.md
│
└── data/
    ├── processed/                  # datasets, manifests, evaluation outputs
    │   └── diamond/                # manual gold annotation assets
    └── resources/                  # normalization lexicon and whitelist
```

## Key Methodology Notes

This part matters if you are reading the repo as a research artifact.

- Historical silver labels were generated on v1 data.
- The active v2 dataset was formed by intersecting v1 silver labels with the v2-cleaned corpus using `review_id`.
- The current gold validation subset is a **single-annotator gold subset**, not a full diamond multi-annotator setup.
- Public GitHub contents intentionally exclude large raw datasets, processed full datasets, and trained model artifacts.

## Quick Start

### Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Training

Baseline sweep:

```powershell
.\scripts\run_baseline_epochs.ps1
```

LoRA sweep:

```powershell
.\scripts\run_lora_epochs.ps1
```

Run both tracks:

```powershell
.\scripts\run_training_experiments.ps1
```

### Run Evaluation

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

### Run Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Main Files to Explore

| File | Purpose |
| --- | --- |
| `src/training/train_baseline.py` | baseline full fine-tuning |
| `src/training/train_lora.py` | parameter-efficient LoRA fine-tuning |
| `src/evaluation/evaluate.py` | experiment comparison and evaluation summary |
| `app.py` | Streamlit interface for live review analysis |
| `scripts/run_baseline_epochs.ps1` | baseline experiment runner |
| `scripts/run_lora_epochs.ps1` | LoRA experiment runner |
| `scripts/build_v2_intersection.py` | build active v2 intersection dataset |
| `scripts/audit_normalization_v2.py` | audit normalization coverage and candidate slang mappings |

## Public Repo Scope

This GitHub version is intentionally lightweight.

Excluded from version control:

- trained model directories
- raw scraped datasets
- full processed research datasets
- local environment files and secrets

Included because they are useful for understanding the work:

- code for the full pipeline
- experiment runners
- methodology notes
- normalization resources
- manifests and annotation support assets

## Portfolio Value

This project demonstrates practical experience in:

- natural language processing
- Indonesian text normalization
- aspect-based sentiment analysis
- weak supervision workflow design
- experiment tracking mindset
- LoRA fine-tuning for transformer models
- Streamlit app prototyping for ML systems

## Next Milestones

- add screenshots and result tables to enrich the public presentation
- add a tiny public sample dataset for demo reproducibility
- complete fresh v2 training comparison
- validate the best model against the manual gold subset
- add a license and lightweight CI checks

Characteristic:
- Gunakan dengan bahasa yang mudah dimengerti, jangan terbelit-belit. Dan straight to the point.
