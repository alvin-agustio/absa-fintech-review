# Model Comparison Assets for Paper

Dokumen ini berisi aset ringkas yang siap ditempel ke paper atau slide.
Versi ini hanya menampilkan model pada epoch 8 agar visual lebih mudah dibaca saat reporting.

## Figure Captions

**Figure 1. Epoch-8 comparison across the LLM-Labelled Validation and LLM-Labelled + Human Subset Validation views, using F1 Macro and F1 Weighted.**
The grouped bar chart compares the four epoch-8 models on two evaluation views and shows both F1 Macro and F1 Weighted.

**Figure 2. Epoch-8 training time comparison between LoRA and non-LoRA models, with LLM and human-subset accuracy annotation.**
The bar chart compares training time at epoch 8 and also annotates both LLM-Labelled Validation accuracy and LLM-Labelled + Human Subset Validation accuracy for each model.

**Figure 3. Training time across the four model families at epochs 3, 5, and 8.**
The grouped bar chart shows how training time grows across epochs for baseline, LoRA, retrained, and retrained LoRA.

**Figure 4. End-to-end funnel from raw reviews to the final model and evaluation set.**
The diagram summarizes the full funnel from raw Google Play reviews, cleaning and normalization, aspect-level dataset creation, clean-subset retraining, and human-subset evaluation.

## Ready-to-use Takeaway

- LLM-Labelled Validation winner at epoch 8: `Retrained LoRA` with Macro F1 `0.8787`.
- LLM-Labelled + Human Subset Validation winner at epoch 8: `LoRA` with Macro F1 `0.8174`.
- Main message: the model that performs best on LLM-Labelled Validation is not the same as the model that performs best on LLM-Labelled + Human Subset Validation.
- Runtime message: LoRA-based models train faster than their non-LoRA counterparts while remaining competitive in accuracy.

## Table

| Family | LLM-Labelled Validation F1 Macro | LLM-Labelled Validation F1 Weighted | LLM-Labelled Validation Accuracy | LLM-Labelled + Human Subset Validation F1 Macro | LLM-Labelled + Human Subset Validation F1 Weighted | LLM-Labelled + Human Subset Validation Accuracy | Training Time (s) | Trainable % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7866 | 0.9521 | 0.9522 | 0.7946 | 0.933 | 0.9323 | 916.2 | - |
| LoRA | 0.7895 | 0.9553 | 0.9567 | 0.8174 | 0.9479 | 0.9522 | 550.81 | 0.47 |
| Retrained | 0.8724 | 0.975 | 0.9753 | 0.8097 | 0.9398 | 0.9442 | 907.52 | - |
| Retrained LoRA | 0.8787 | 0.978 | 0.9784 | 0.7076 | 0.9292 | 0.9402 | 539.55 | 0.47 |

## Files

- `model_comparison_epoch8_benchmarks.png`
- `model_comparison_epoch8_training_time.png`
- `model_comparison_training_time_all_epochs.png`
- `model_building_pipeline_end_to_end.png`
- `model_comparison_epoch8_table.csv`