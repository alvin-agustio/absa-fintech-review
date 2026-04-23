# Model Comparison Assets for Paper

Dokumen ini berisi aset ringkas yang siap ditempel ke paper atau slide.
Versi ini menggabungkan perbandingan epoch 8 untuk metrik validasi dan best-point comparison untuk grafik training time.

## Figure Captions

**Figure 1. Epoch-8 comparison across the LLM-Labelled Validation and LLM-Labelled + Human Subset Validation views, using F1 Macro and F1 Weighted.**
The grouped bar chart compares the epoch-8 model families on two evaluation views and shows both F1 Macro and F1 Weighted.

**Figure 2. Training time at the best weak-label validation point across model families, annotated with the latest available gold Macro-F1.**
The bar chart compares training time at the best weak-label Macro-F1 point for each family and annotates each family with that weak-label Macro-F1 plus the latest available gold Macro-F1 from the final evaluated run.

**Figure 3. Training time across model families at epochs 3, 5, and 8.**
The grouped bar chart shows how training time grows across the available model families at epochs 3, 5, and 8.

**Figure 4. End-to-end funnel from raw reviews to the final model and evaluation set.**
The diagram summarizes the full funnel from raw Google Play reviews, cleaning and normalization, aspect-level dataset creation, clean-subset retraining, and human-subset evaluation.

**Figure 5. Agreement between the LLM-labelled validation set and the manual human subset.**
The scatter plot contrasts the best LLM-labelled Macro-F1 point for each family against the latest available human-subset Macro-F1, highlighting models that generalize well and those that overfit the weak labels.

**Figure 5a. Best-point grouped bar chart across the LLM-labelled and human-subset views.**
The grouped bar chart compares each family's best LLM-labelled Macro-F1 against its latest available human-subset Macro-F1, with the best LLM epoch shown above each bar.

**Figure 6. Before-versus-after retraining slope chart.**
The slope chart compares each base family with its retrained counterpart on both the LLM-labelled validation set and the manual human subset.

**Figure 6a. Retraining delta bar chart.**
The delta bar chart shows how much each family changes after retraining on the LLM-labelled validation set and on the manual human subset.

**Figure 7. Training-cost versus validation-quality trade-off.**
The two-panel bubble chart compares training time at each family's best LLM-labelled point against both LLM-labelled Macro-F1 and human-subset Macro-F1, while bubble size reflects trainable parameter share.

**Figure 8. Generalization gap from LLM-labelled to human-subset validation.**
The bar chart shows how far each family drops when moving from the weak-label setting to the manual human-subset setting.

**Figure 9. Uncertainty, noise, and outcome heatmap.**
The heatmap combines MC-dropout entropy, variance, mismatch rate, noise ratio, and final human-subset Macro-F1 for the families that have uncertainty artifacts.

**Figure 10. Aspect-level noise ratio heatmap.**
The heatmap breaks the detected noise ratio into the service, risk, and trust aspects for the families with noise-detection artifacts.

**Figure 11. End-to-end short-diagnosis logic.**
The flow diagram explains how the dashboard moves from raw review text to ABSA sentiment prediction, aspect-presence filtering, issue taxonomy assignment, evidence ranking, and final short diagnosis cards.

**Figure 12. Complete rule-based taxonomy across all aspects.**
The taxonomy map lists every issue bucket for risk, trust, and service, together with its intent and example keywords.

**Appendix CSV. Complete taxonomy table across all aspects.**
The CSV version lists every aspect, bucket, priority, description, and keyword set in a tabular form for audit and appendix use.

## Ready-to-use Takeaway

- LLM-Labelled Validation winner at epoch 8: `Retrained AdaLoRA` with Macro F1 `0.8747`.
- LLM-Labelled + Human Subset Validation winner at epoch 8: `Retrained` with Macro F1 `0.7681`.
- Main message: the model that performs best on LLM-Labelled Validation is not the same as the model that performs best on LLM-Labelled + Human Subset Validation.
- Runtime message: standard PEFT families remain the fastest, while QLoRA-based families are the slowest in the latest run set.
- Practical message: the most useful operational comparison is the trade-off between training time and human-subset quality, not the raw epoch-by-epoch progression alone.

## Table

| Family | LLM-Labelled Validation F1 Macro | LLM-Labelled Validation F1 Weighted | LLM-Labelled Validation Accuracy | LLM-Labelled + Human Subset Validation F1 Macro | LLM-Labelled + Human Subset Validation F1 Weighted | LLM-Labelled + Human Subset Validation Accuracy | Training Time (s) | Trainable % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7824 | 0.9498 | 0.95 | 0.7284 | 0.9269 | 0.9323 | 2109.1 | - |
| LoRA | 0.7916 | 0.9549 | 0.9558 | 0.7531 | 0.9359 | 0.9442 | 1425.9 | 0.47 |
| DoRA | 0.7876 | 0.9531 | 0.9547 | 0.7531 | 0.9359 | 0.9442 | 1813.63 | 0.49 |
| AdaLoRA | 0.7575 | 0.9475 | 0.9511 | 0.7136 | 0.9334 | 0.9442 | 2436.97 | 0.36 |
| QLoRA | 0.7558 | 0.9459 | 0.9495 | 0.6114 | 0.9195 | 0.9402 | 4451.97 | 0.72 |
| Retrained | 0.8189 | 0.9591 | 0.9604 | 0.7681 | 0.936 | 0.9402 | 2090.94 | - |
| Retrained LoRA | 0.8021 | 0.9686 | 0.971 | 0.6105 | 0.9175 | 0.9363 | 1397.41 | 0.47 |
| Retrained DoRA | 0.8105 | 0.9666 | 0.9664 | 0.6076 | 0.9131 | 0.9323 | 1778.55 | 0.49 |
| Retrained AdaLoRA | 0.8747 | 0.9802 | 0.9804 | 0.6144 | 0.9193 | 0.9402 | 2343.89 | 0.36 |
| Retrained QLoRA | 0.8257 | 0.9827 | 0.984 | 0.6114 | 0.9195 | 0.9402 | 4323.58 | 0.72 |

## Files

- `model_comparison_epoch8_benchmarks.png`
- `model_comparison_epoch8_training_time.png`
- `model_comparison_training_time_all_epochs.png`
- `model_building_pipeline_end_to_end.png`
- `model_agreement_llm_vs_human_best_point.png`
- `model_best_point_llm_vs_human_bar.png`
- `model_retraining_delta_slope.png`
- `model_retraining_delta_bar.png`
- `model_tradeoff_time_vs_human_f1.png`
- `model_generalization_gap_best_point.png`
- `model_uncertainty_noise_heatmap.png`
- `model_aspect_noise_heatmap.png`
- `diagnosis_short_logic_end_to_end.png`
- `taxonomy_complete_all_aspects.png`
- `taxonomy_complete_all_aspects.csv`
- `model_comparison_epoch8_table.csv`