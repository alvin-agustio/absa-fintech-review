# Model Comparison Assets for Paper

Dokumen ini berisi aset yang siap ditempel ke paper atau slide.

## Figure Captions

**Figure 1. Weak-label benchmark across model families and training epochs.**
The grouped bar chart compares Macro F1 on the weak-label test set for four model families evaluated at 3, 5, and 8 epochs.

**Figure 2. Gold-subset benchmark across model families and training epochs.**
The grouped bar chart compares Macro F1 on the manually annotated gold subset for the same four model families and three epoch settings.

## Ready-to-use Takeaway

- Weak-label winner: `Retrained LoRA` epoch `8` with Macro F1 `0.8787`.
- Gold-subset winner: `LoRA` epoch `8` with Macro F1 `0.8174`.
- Main message: the model that performs best on weak labels is not the same as the model that performs best on the human gold subset.

## Table

| Family | Epoch | Weak F1 Macro | Weak Accuracy | Gold F1 Macro | Gold Accuracy | Training Time (s) | Trainable % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 3 | 0.7805 | 0.957 | 0.706 | 0.9363 | 347.15 | - |
| Baseline | 5 | 0.7933 | 0.9541 | 0.7901 | 0.9402 | 568.39 | - |
| Baseline | 8 | 0.7866 | 0.9522 | 0.7946 | 0.9323 | 916.2 | - |
| LoRA | 3 | 0.7799 | 0.9573 | 0.6144 | 0.9402 | 205.17 | 0.47 |
| LoRA | 5 | 0.7849 | 0.9571 | 0.7477 | 0.9442 | 341.53 | 0.47 |
| LoRA | 8 | 0.7895 | 0.9567 | 0.8174 | 0.9522 | 550.81 | 0.47 |
| Retrained | 3 | 0.862 | 0.9758 | 0.7436 | 0.9402 | 344.72 | - |
| Retrained | 5 | 0.8644 | 0.9761 | 0.731 | 0.9323 | 589.73 | - |
| Retrained | 8 | 0.8724 | 0.9753 | 0.8097 | 0.9442 | 907.52 | - |
| Retrained LoRA | 3 | 0.8459 | 0.9737 | 0.6098 | 0.9363 | 203.03 | 0.47 |
| Retrained LoRA | 5 | 0.8715 | 0.9768 | 0.6657 | 0.9402 | 449.84 | 0.47 |
| Retrained LoRA | 8 | 0.8787 | 0.9784 | 0.7076 | 0.9402 | 539.55 | 0.47 |

## Files

- `model_comparison_weak.png`
- `model_comparison_gold.png`
- `model_comparison_table.csv`