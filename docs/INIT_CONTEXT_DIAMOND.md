# INIT CONTEXT: DIAMOND FORMAT

This file is the high-context bootstrap for continuing work in this repository without losing research, engineering, and product context.

Use this as the first file to read in future sessions.

---

## D — Domain, Direction, and Dissertation Scope

### Project identity

- Project name: `Fintech Review ABSA`
- Repository role:
  - active thesis / skripsi workspace
  - public portfolio-grade NLP / ML engineering project
- Core task:
  - Aspect-Based Sentiment Analysis on Indonesian Google Play reviews
- Domain:
  - fintech lending apps in Indonesia
- Target apps:
  - `Kredivo` → `com.finaccel.android`
  - `Akulaku` → `io.silvrr.installment`

### Research scope

This repository studies how to perform ABSA on noisy Indonesian fintech app reviews using IndoBERT, then compare:

- baseline full fine-tuning
- LoRA fine-tuning
- uncertainty-aware filtering
- retraining on cleaner subset

and finally surface the results in a dashboard.

### Thesis-aligned research questions

The practical interpretation of the current research direction is:

1. How IndoBERT can be applied to ABSA for fintech lending reviews.
2. How LoRA and uncertainty-aware processing affect model performance.
3. How the results can be implemented into a dashboard that visualizes user perception.

### Final task framing

The repo is no longer just “train a model and compare scores”.

It now has three layers:

1. **Modeling layer**
   - baseline, LoRA, retrained, retrained LoRA
2. **Validation layer**
   - weak-label evaluation
   - human-annotated gold subset evaluation
3. **Presentation layer**
   - Streamlit observatory/dashboard

---

## I — Inventory of the Repository

### Top-level structure

```text
.
├── app.py
├── config.py
├── requirements.txt
├── docs/
├── data/
├── droplet/
├── models/
├── scripts/
└── src/
```

### Most important files by purpose

#### Core configuration

- `config.py`
  - central source of truth for paths, apps, label config, model config

#### Data pipeline

- `src/data/scrape_reviews.py`
  - Google Play scraping pipeline
- `src/data/preprocess.py`
  - cleaning / normalization of raw review text
- `src/data/labeling.py`
  - LLM-based silver labeling workflow

#### Training

- `src/training/train_baseline.py`
  - full fine-tuning
- `src/training/train_lora.py`
  - LoRA training on weak-label data
- `src/training/retrain_filtered.py`
  - full fine-tuning on clean subset
- `src/training/train_lora_filtered.py`
  - LoRA training on clean subset

#### Evaluation

- `src/evaluation/evaluate.py`
  - main experiment summary and epoch comparison
- `src/evaluation/predict_mc_dropout.py`
  - MC Dropout uncertainty estimation
- `src/evaluation/detect_label_noise.py`
  - weak-label noise detection
- `src/evaluation/evaluate_gold_subset.py`
  - evaluation on manually annotated gold subset

#### Dashboard

- `app.py`
  - current Streamlit observatory entrypoint
- `src/dashboard/storage.py`
  - local dashboard store and cache
- `src/dashboard/registry.py`
  - experiment registry built from artifacts
- `src/dashboard/live.py`
  - live review fetch + inference orchestration
- `src/dashboard/analytics.py`
  - KPI/trend/evidence analytics
- `src/dashboard/research.py`
  - gold subset / model compare / uncertainty lab support

#### Execution scripts

- `scripts/run_baseline_epochs.ps1`
- `scripts/run_lora_epochs.ps1`
- `scripts/run_training_experiments.ps1`
- `scripts/setup_digitalocean_gpu.sh`

#### Annotation / diamond assets

- `data/processed/diamond/template_anotator_tunggal_balanced_300_aspect_rows.csv`
- `data/processed/diamond/pedoman_anotasi_diamond.md`
- `data/processed/diamond/evaluation_all_models/gold_evaluation_overview.csv`
- `data/processed/diamond/evaluation_all_models/gold_evaluation_summary.json`

#### Droplet artifacts

- `droplet/skripsi_eval_core/data/processed/evaluation/evaluation_summary.json`
- `droplet/skripsi_eval_core/data/processed/evaluation/epoch_comparison_summary.csv`
- `droplet/skripsi_eval_core/data/processed/noise/noise_summary.json`
- `droplet/skripsi_eval_core/data/processed/uncertainty/mc_summary.json`

---

## A — Architecture and Data Flow

### End-to-end pipeline

```text
Google Play reviews
  -> scraping
  -> preprocessing / normalization
  -> silver / weak labels
  -> v2 intersection training set
  -> model training (baseline / LoRA)
  -> uncertainty estimation (MC Dropout)
  -> label noise detection
  -> clean subset creation
  -> retraining (full FT / LoRA)
  -> evaluation on weak-label split
  -> evaluation on gold subset
  -> dashboard / observatory
```

### Current dataset sources in config

From `config.py`:

- official training dataset:
  - `data/processed/dataset_absa_50k_v2_intersection.csv`
- training manifest:
  - `data/processed/manifests/stratified_50k_seed42_v2_intersection.csv`
- active gold template:
  - `data/processed/diamond/template_anotator_tunggal_balanced_300_aspect_rows.csv`

### Label space

- aspects:
  - `risk`
  - `trust`
  - `service`
- sentiments:
  - `Negative`
  - `Neutral`
  - `Positive`

### Model backbone

- backbone:
  - `indobenchmark/indobert-base-p1`
- max length:
  - `128`
- LoRA config:
  - `r = 16`
  - `alpha = 32`
  - `dropout = 0.1`
  - target modules:
    - `query`
    - `value`

### Dashboard architecture

The dashboard is no longer the original simple demo. It now uses a service-layer pattern:

- `app.py`
  - UI shell + navigation
- `src/dashboard/storage.py`
  - local analytics persistence
  - DuckDB if available
  - sqlite fallback if DuckDB not installed
- `src/dashboard/registry.py`
  - registry of available models and rankings
- `src/dashboard/live.py`
  - live fetch -> preprocess -> infer -> persist
- `src/dashboard/analytics.py`
  - all scope-level metrics and review-level evidence shaping
- `src/dashboard/research.py`
  - gold subset, hardest cases, absent-aspect tendency, model ladder

### Dashboard sections

Current app sections:

1. `Command Center`
2. `Live Analysis`
3. `Evidence Explorer`
4. `Research Lab`

### Dashboard storage paths

- database:
  - `data/processed/dashboard/dashboard.duckdb`
- cache folder:
  - `data/processed/dashboard/cache/`

### Live Analysis behavior

Live analysis does:

1. choose app scope
2. choose date window
3. choose model
4. fetch Google Play reviews on demand
5. preprocess text
6. run ABSA inference
7. persist job cache locally
8. reuse cache if the same scope is requested again

Important behavior:

- requested review limit is not guaranteed as final output count
- final count may be smaller due to:
  - source availability from Google Play
  - date filtering
  - deduplication
  - removal of short texts

---

## M — Methodology, Experiment History, and Annotation Logic

### Training tracks that exist

There are four experiment families:

1. `baseline`
   - full fine-tuning on weak-label dataset
2. `lora`
   - LoRA on weak-label dataset
3. `retrained`
   - full fine-tuning on clean subset after uncertainty-aware filtering
4. `retrained_lora`
   - LoRA on clean subset after uncertainty-aware filtering

Each family has been run at:

- epoch `3`
- epoch `5`
- epoch `8`

### Weak-label / uncertainty-aware methodology

The uncertainty-aware branch is:

1. choose baseline model
2. run MC Dropout over weak-label dataset
3. compute uncertainty metrics
4. detect likely noisy labels
5. separate:
   - clean subset
   - noisy candidates
6. retrain on clean subset

### Noise filtering summary

From `droplet/skripsi_eval_core/data/processed/evaluation/evaluation_summary.json`:

- `n_total = 53366`
- `n_noisy_candidates = 1117`
- `n_clean = 52249`
- `noise_ratio = 0.020930929805494134`
- uncertainty metric used:
  - `uncertainty_entropy`
- threshold:
  - `0.008582847`

### Uncertainty summary

From the same summary:

- `num_mc = 30`
- `mean_entropy = 0.024171428754925728`
- `mean_variance = 0.001047299476340413`
- `error_rate_vs_weak = 0.022860997638946147`

### Gold subset methodology

The current gold subset is manual aspect-level evaluation data.

Current actual file:

- `data/processed/diamond/template_anotator_tunggal_balanced_300_aspect_rows.csv`

Actual current status:

- total rows: `300`
- aspect distribution:
  - `service = 100`
  - `risk = 100`
  - `trust = 100`
- `aspect_present = 1`:
  - `251`
- `aspect_present = 0`:
  - `49`
- label distribution:
  - `Negative = 204`
  - `Positive = 36`
  - `Neutral = 11`
  - blank label for absent rows = `49`

### Important annotation rule change

This is critical future-session context:

- earlier discussion considered:
  - `aspect_present = 0` -> `label = Neutral`
- but the current actual dataset was updated to:
  - `aspect_present = 0` -> `label = blank`

This means:

- the **gold CSV is the actual source of truth**
- the old prose guideline in `pedoman_anotasi_diamond.md` is partly outdated on this point

### Why this matters

The current ABSA task should be interpreted as:

- sentiment classification is meaningful only when `aspect_present = 1`
- absence handling is diagnostic / methodological, not the main public sentiment score

### Gold evaluation interpretation

Manual gold evaluation is used as a stronger human validation layer, not as training data for the main experiment branch.

This is one of the biggest methodological strengths of the repo.

---

## O — Outputs, Results, and Current Findings

### Weak-label epoch sweep summary

From `droplet/skripsi_eval_core/data/processed/evaluation/epoch_comparison_summary.csv`:

| Model | Epoch | Accuracy | F1 Macro | Training Time (s) |
| --- | --- | ---: | ---: | ---: |
| baseline | 3 | 0.9570 | 0.7805 | 347.15 |
| lora | 3 | 0.9573 | 0.7799 | 205.17 |
| retrained | 3 | 0.9758 | 0.8620 | 344.72 |
| retrained_lora | 3 | 0.9737 | 0.8459 | 203.03 |
| baseline | 5 | 0.9541 | 0.7933 | 568.39 |
| lora | 5 | 0.9571 | 0.7849 | 341.53 |
| retrained | 5 | 0.9761 | 0.8644 | 589.73 |
| retrained_lora | 5 | 0.9768 | 0.8715 | 449.84 |
| baseline | 8 | 0.9522 | 0.7866 | 916.20 |
| lora | 8 | 0.9567 | 0.7895 | 550.81 |
| retrained | 8 | 0.9753 | 0.8724 | 907.52 |
| retrained_lora | 8 | 0.9784 | 0.8787 | 539.55 |

### Weak-label best model

On weak-label / silver-style evaluation, best overall was:

- `retrained_lora_epoch8`
- `accuracy = 0.9784`
- `f1_macro = 0.8787`
- `f1_weighted = 0.9780`

### Gold subset ranking

From `data/processed/diamond/evaluation_all_models/gold_evaluation_overview.csv`:

Top models on human gold subset:

1. `lora_epoch8`
   - `accuracy = 0.9522`
   - `f1_macro = 0.8174`
2. `retrained_epoch8`
   - `accuracy = 0.9442`
   - `f1_macro = 0.8097`
3. `baseline_epoch8`
   - `accuracy = 0.9323`
   - `f1_macro = 0.7946`
4. `baseline_epoch5`
   - `accuracy = 0.9402`
   - `f1_macro = 0.7901`

### Important research conclusion

This is one of the most important high-level findings:

- weak-label winner:
  - `retrained_lora_epoch8`
- gold-subset winner:
  - `lora_epoch8`

Therefore:

- the best model under weak-label evaluation is **not automatically** the best model under human validation
- uncertainty-aware filtering is useful, but not a guaranteed final winner under gold evaluation

### Interpreting the uncertainty-aware pipeline

The honest interpretation is:

- uncertainty-aware processing is **relevant and useful**
- it improves weak-label-side performance strongly
- but it does **not automatically dominate** in manual gold evaluation

This does **not** mean the approach failed.
It means:

- weak-label improvement and human-validity improvement are related but not identical objectives

### Main gold subset error patterns

Observed recurring error themes:

1. multi-aspect reviews
2. aspect assignment drift
3. overly negative collapse
4. mixed sentiment cases
5. question / ambiguity / speculative language
6. sarcasm / hyperbole / rough wording
7. aspect-specific sentiment vs global review tone mismatch
8. absent or weak aspect presence

### Presence-related interpretation

Presence-related metrics in gold evaluation are currently diagnostic only.

Why:

- main models are sentiment classifiers conditioned on aspect
- they are not pure `aspect_present` classifiers

So:

- ECE / confidence interpretation is valid mainly for sentiment prediction when aspect is present
- not as a clean primary metric for aspect-presence detection

---

## N — Next Steps, Open Work, and Practical Continuation

### Highest-priority next work

The current project is no longer blocked on core modeling.

Most useful next steps are:

1. stabilize and polish the dashboard UX
2. tighten dashboard diagnostics
3. finalize thesis writing:
   - results
   - discussion
   - limitations
   - recommendations

### Dashboard-specific next work

Current dashboard has strong foundations, but still needs polish.

Recommended next tasks:

1. improve UI spacing / grouping
2. remove any remaining layout artifacts from Streamlit + raw HTML mixing
3. add fetch diagnostics:
   - raw fetched
   - in-range
   - deduplicated
   - removed short
   - final kept
4. allow custom review limit beyond presets
5. refine compare lens and evidence drill-down

### Modeling next work

If more modeling is done, the most meaningful extension would be:

- explicit `aspect_present` classifier as a separate task

But this is optional and should be framed as an extension, not a requirement to close current research.

### Writing next work

The strongest narrative now is:

1. weak-label pipeline built at scale
2. LoRA and full fine-tuning compared
3. uncertainty-aware filtering added
4. retraining improved weak-label performance
5. gold subset revealed that human ranking can differ from weak-label ranking
6. dashboard operationalizes the result

### Practical “ready to demo” checklist

Before final presentation:

1. install all dashboard dependencies
2. run Streamlit without layout regressions
3. pre-load at least one live analysis job cache
4. verify gold model ranking renders correctly
5. verify research lab shows:
   - gold ranking
   - model ladder
   - uncertainty summary
   - hardest cases

---

## D — Decisions, Defaults, Gotchas, and Do-Not-Forget Notes

### Current defaults that matter

- default dashboard model should follow best gold-subset model
- live fetch uses Google Play newest reviews
- live results are cached locally by:
  - app scope
  - date window
  - review limit
  - model id

### Dependency gotchas

- `duckdb` is optional in code path, but preferred for dashboard storage
  - sqlite fallback exists
- `google_play_scraper` is lazily imported
  - dashboard can open without it
  - but `Live Analysis` needs it

### UI gotcha

Streamlit widgets do not truly nest inside arbitrary raw HTML `<div>` blocks.

This matters because:

- pure HTML wrappers may render as empty styled boxes
- widgets may appear visually outside intended cards

Current direction:

- use HTML only for lightweight typography blocks
- keep widget layouts mostly Streamlit-native

### Gold subset source of truth

Use the CSV, not assumptions.

Current true source:

- `data/processed/diamond/template_anotator_tunggal_balanced_300_aspect_rows.csv`

Important current truth:

- absent rows have blank `label`
- do not overwrite this assumption from older notes without explicit review

### Evaluation source of truth

For final comparative claims, use:

- weak-label / silver-side:
  - `droplet/skripsi_eval_core/data/processed/evaluation/epoch_comparison_summary.csv`
- gold-side:
  - `data/processed/diamond/evaluation_all_models/gold_evaluation_overview.csv`

### Final current model interpretation

There are two “best” notions:

1. best on weak-label evaluation:
   - `retrained_lora_epoch8`
2. best on human gold evaluation:
   - `lora_epoch8`

Any future summary must state which benchmark is being used.

### Safe continuation commands

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run dashboard:

```powershell
streamlit run app.py
```

Run main evaluation:

```powershell
python src/evaluation/evaluate.py
```

Run gold subset evaluation:

```powershell
python src/evaluation/evaluate_gold_subset.py
```

### If a future session starts cold

Read in this order:

1. `docs/INIT_CONTEXT_DIAMOND.md`
2. `config.py`
3. `README.md`
4. `app.py`
5. `data/processed/diamond/evaluation_all_models/gold_evaluation_overview.csv`
6. `droplet/skripsi_eval_core/data/processed/evaluation/epoch_comparison_summary.csv`

---

## Quick TL;DR

- This repo is an Indonesian fintech ABSA thesis + portfolio project.
- The main backbone is IndoBERT with baseline, LoRA, retrained, and retrained LoRA tracks.
- Uncertainty-aware filtering produced a cleaner subset and improved weak-label-side metrics.
- Weak-label best model is `retrained_lora_epoch8`.
- Human gold-subset best model is `lora_epoch8`.
- The project now has a richer Streamlit observatory, not just a simple demo app.
- The most important remaining work is polish, diagnostics, and final narrative writing, not rebuilding the experiment core.
