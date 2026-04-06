from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
WEAK_CSV = ROOT / "droplet" / "skripsi_eval_core" / "data" / "processed" / "evaluation" / "epoch_comparison_summary.csv"
GOLD_CSV = ROOT / "data" / "processed" / "diamond" / "evaluation_all_models" / "gold_evaluation_overview.csv"
V2_REPORT_JSON = ROOT / "data" / "processed" / "dataset_absa_v2_report.json"
INTERSECTION_REPORT_JSON = ROOT / "data" / "processed" / "manifests" / "stratified_50k_seed42_v2_intersection_report.json"
NOISE_REPORT_JSON = ROOT / "droplet" / "skripsi_eval_core" / "data" / "processed" / "noise" / "noise_summary.json"
OUT_DIR = ROOT / "docs" / "paper_assets"
EPOCH8_COMPARISON_PNG = "model_comparison_epoch8_benchmarks.png"
TRAINING_TIME_PNG = "model_comparison_epoch8_training_time.png"
TRAINING_TIME_ALL_EPOCHS_PNG = "model_comparison_training_time_all_epochs.png"
PIPELINE_FUNNEL_PNG = "model_building_pipeline_end_to_end.png"
TABLE_CSV = "model_comparison_epoch8_table.csv"
RAW_REVIEW_ROWS = 505936

FAMILY_ORDER = ["baseline", "lora", "retrained", "retrained_lora"]
FAMILY_LABELS = {
    "baseline": "Baseline",
    "lora": "LoRA",
    "retrained": "Retrained",
    "retrained_lora": "Retrained LoRA",
}
BENCHMARK_COLORS = {
    "LLM-Labelled Validation F1 Macro": "#B44E34",
    "LLM-Labelled + Human Subset Validation F1 Macro": "#1B7286",
}
TIME_COLORS = {
    "non_lora": "#B8BDC7",
    "lora": "#7FA37C",
}
EPOCH_COLORS = {
    3: "#C9CED6",
    5: "#7FA37C",
    8: "#B44E34",
}


def load_combined_frame() -> pd.DataFrame:
    weak = pd.read_csv(WEAK_CSV)
    gold = pd.read_csv(GOLD_CSV)

    weak = weak.rename(
        columns={
            "model": "family",
            "accuracy": "weak_accuracy",
            "f1_macro": "weak_f1_macro",
            "f1_weighted": "weak_f1_weighted",
        }
    )
    weak = weak[
        [
            "family",
            "epochs",
            "weak_accuracy",
            "weak_f1_macro",
            "weak_f1_weighted",
            "training_time_seconds",
            "trainable_pct",
        ]
    ].copy()

    gold["family"] = gold["model_name"].str.replace(r"_epoch\d+$", "", regex=True)
    gold["epochs"] = gold["model_name"].str.extract(r"epoch(\d+)").astype(int)
    gold = gold.rename(
        columns={
            "sentiment_accuracy_present": "gold_accuracy",
            "sentiment_f1_macro_present": "gold_f1_macro",
            "sentiment_f1_weighted_present": "gold_f1_weighted",
        }
    )
    gold = gold[
        ["family", "epochs", "gold_accuracy", "gold_f1_macro", "gold_f1_weighted", "model_type"]
    ].copy()

    combined = weak.merge(gold, on=["family", "epochs"], how="inner")
    combined["family"] = pd.Categorical(combined["family"], categories=FAMILY_ORDER, ordered=True)
    combined = combined.sort_values(["family", "epochs"]).reset_index(drop=True)
    combined["family_label"] = combined["family"].map(FAMILY_LABELS)
    combined["model_name"] = combined["family"].astype(str) + "_epoch" + combined["epochs"].astype(str)
    return combined


def load_pipeline_numbers() -> dict[str, int]:
    v2_report = json.loads(V2_REPORT_JSON.read_text(encoding="utf-8"))
    intersection_report = json.loads(INTERSECTION_REPORT_JSON.read_text(encoding="utf-8"))
    noise_report = json.loads(NOISE_REPORT_JSON.read_text(encoding="utf-8"))
    return {
        "raw_reviews": RAW_REVIEW_ROWS,
        "clean_v1": int(v2_report["reviews_clean_v1_rows"]),
        "clean_v2": int(v2_report["reviews_clean_v2_rows"]),
        "cohort_reviews": int(intersection_report["intersection_rows"]),
        "labeled_reviews": int(intersection_report["v2_intersection_labeled_any"]),
        "aspect_rows": int(noise_report["n_total"]),
        "clean_subset": int(noise_report["n_clean"]),
        "human_subset": 300,
        "trained_models": 12,
    }


def plot_epoch8_benchmark_bar(df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), dpi=200, sharey=False)

    family_labels = [FAMILY_LABELS[item] for item in FAMILY_ORDER]
    x = list(range(len(FAMILY_ORDER)))
    width = 0.34
    offsets = {
        "LLM-Labelled Validation": -width / 2,
        "LLM-Labelled + Human Subset Validation": width / 2,
    }
    panel_specs = [
        (
            axes[0],
            "F1 Macro",
            [
                ("LLM-Labelled Validation", "LLM-Labelled Validation F1 Macro"),
                ("LLM-Labelled + Human Subset Validation", "LLM-Labelled + Human Subset Validation F1 Macro"),
            ],
        ),
        (
            axes[1],
            "F1 Weighted",
            [
                ("LLM-Labelled Validation", "LLM-Labelled Validation F1 Weighted"),
                ("LLM-Labelled + Human Subset Validation", "LLM-Labelled + Human Subset Validation F1 Weighted"),
            ],
        ),
    ]

    for ax, panel_title, metrics in panel_specs:
        for display_label, column_name in metrics:
            values = df.set_index("Family").reindex(family_labels)[column_name].tolist()
            bars = ax.bar(
                [pos + offsets[display_label] for pos in x],
                values,
                width=width,
                label=display_label,
                color=BENCHMARK_COLORS[column_name.replace(" F1 Weighted", " F1 Macro").replace(" F1 Macro", " F1 Macro")],
                edgecolor="white",
                linewidth=0.8,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.004,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#30333A",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(family_labels, fontsize=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(panel_title, fontsize=11, weight="bold")

    axes[0].set_ylim(0.68, 0.90)
    axes[1].set_ylim(0.90, 0.985)
    axes[0].set_ylabel("Score", fontsize=10)
    axes[0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(1.05, 1.20))
    fig.suptitle(title, fontsize=13, weight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_training_time_bar(df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=200)

    family_order = ["Baseline", "LoRA", "Retrained", "Retrained LoRA"]
    plot_df = df.set_index("Family").reindex(family_order).reset_index()
    colors = [
        TIME_COLORS["lora"] if "LoRA" in family else TIME_COLORS["non_lora"]
        for family in plot_df["Family"]
    ]

    bars = ax.bar(
        plot_df["Family"],
        plot_df["Training Time (s)"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.58,
    )

    max_value = float(plot_df["Training Time (s)"].max())
    offset = max_value * 0.035
    for bar, value, llm_acc, human_acc in zip(
        bars,
        plot_df["Training Time (s)"],
        plot_df["LLM-Labelled Validation Accuracy"],
        plot_df["LLM-Labelled + Human Subset Validation Accuracy"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + offset,
            f"{float(value):.0f}s\nLLM {float(llm_acc):.3f} | Human {float(human_acc):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#30333A",
        )

    ax.set_ylabel("Training Time (seconds)", fontsize=10)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylim(0, max_value * 1.18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIME_COLORS["non_lora"]),
        plt.Rectangle((0, 0), 1, 1, color=TIME_COLORS["lora"]),
    ]
    ax.legend(
        handles,
        ["Non-LoRA", "LoRA"],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_training_time_all_epochs(df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=200)

    family_labels = [FAMILY_LABELS[item] for item in FAMILY_ORDER]
    x = list(range(len(FAMILY_ORDER)))
    width = 0.22
    offsets = {3: -width, 5: 0.0, 8: width}

    max_value = float(df["training_time_seconds"].max())
    label_offset = max_value * 0.02

    for epoch in [3, 5, 8]:
        subset = df[df["epochs"] == epoch].set_index("family").reindex(FAMILY_ORDER)
        values = subset["training_time_seconds"].tolist()
        bars = ax.bar(
            [pos + offsets[epoch] for pos in x],
            values,
            width=width,
            label=f"Epoch {epoch}",
            color=EPOCH_COLORS[epoch],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + label_offset,
                f"{float(value):.0f}s",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#30333A",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(family_labels, fontsize=10)
    ax.set_ylabel("Training Time (seconds)", fontsize=10)
    ax.set_ylim(0, max_value * 1.18)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_pipeline_funnel(numbers: dict[str, int], out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9.5, 10.8), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        ("Review mentah Google Play", f"{numbers['raw_reviews']:,} ulasan", "Data awal dari Akulaku dan Kredivo"),
        ("Setelah pembersihan dasar", f"{numbers['clean_v1']:,} ulasan", "Duplikat, review kosong, dan review terlalu pendek dibuang"),
        ("Setelah normalisasi final", f"{numbers['clean_v2']:,} ulasan", "Teks dirapikan agar lebih konsisten untuk proses modelling"),
        ("Cohort eksperimen resmi", f"{numbers['cohort_reviews']:,} ulasan", "Kumpulan review-level yang dipakai untuk eksperimen utama"),
        ("Review yang punya label aspek", f"{numbers['labeled_reviews']:,} ulasan", "Hanya review dengan minimal satu label risk, trust, atau service"),
        ("Data training level-aspek", f"{numbers['aspect_rows']:,} baris", "Setiap review dipecah menjadi pasangan review-aspect untuk training"),
        ("Clean subset setelah filtering", f"{numbers['clean_subset']:,} baris", "Baris yang diduga noisy dibuang, sisanya dipakai untuk retraining"),
        ("Pelatihan model", f"{numbers['trained_models']} model", "4 variasi model x 3 epoch = 12 training runs"),
        ("Evaluasi human subset", f"{numbers['human_subset']} baris", "Subset manual dipakai untuk memilih model yang paling masuk akal bagi manusia"),
    ]

    y = 0.93
    box_h = 0.08
    gap = 0.022
    for idx, (title, value, desc) in enumerate(stages):
        rect = FancyBboxPatch(
            (0.1, y - box_h),
            0.8,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#d8d2c6",
            facecolor="#fdfbf6" if idx % 2 == 0 else "#f7f2e8",
        )
        ax.add_patch(rect)
        ax.text(0.13, y - 0.028, title, fontsize=10.5, fontweight="bold", color="#1f211e", va="center")
        ax.text(0.13, y - 0.055, desc, fontsize=8.6, color="#50544e", va="center")
        ax.text(0.86, y - 0.04, value, fontsize=10, fontweight="bold", color="#b44e34", ha="right", va="center")

        if idx < len(stages) - 1:
            arrow = FancyArrowPatch(
                (0.5, y - box_h - 0.005),
                (0.5, y - box_h - gap + 0.005),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
                color="#8c8f95",
            )
            ax.add_patch(arrow)
        y -= box_h + gap

    ax.text(0.5, 0.985, "Funneling Data dari Review Mentah sampai Model Jadi", ha="center", va="top", fontsize=14, fontweight="bold", color="#1f211e")
    ax.text(0.5, 0.963, "Alur singkat dari preprocessing, labeling, training, sampai evaluasi akhir", ha="center", va="top", fontsize=9.5, color="#50544e")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def save_table(df: pd.DataFrame) -> pd.DataFrame:
    table_df = df[df["epochs"] == 8][
        [
            "family_label",
            "weak_f1_macro",
            "weak_f1_weighted",
            "weak_accuracy",
            "gold_f1_macro",
            "gold_f1_weighted",
            "gold_accuracy",
            "training_time_seconds",
            "trainable_pct",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "family_label": "Family",
            "weak_f1_macro": "LLM-Labelled Validation F1 Macro",
            "weak_f1_weighted": "LLM-Labelled Validation F1 Weighted",
            "weak_accuracy": "LLM-Labelled Validation Accuracy",
            "gold_f1_macro": "LLM-Labelled + Human Subset Validation F1 Macro",
            "gold_f1_weighted": "LLM-Labelled + Human Subset Validation F1 Weighted",
            "gold_accuracy": "LLM-Labelled + Human Subset Validation Accuracy",
            "training_time_seconds": "Training Time (s)",
            "trainable_pct": "Trainable %",
        }
    )
    for col in [
        "LLM-Labelled Validation F1 Macro",
        "LLM-Labelled Validation F1 Weighted",
        "LLM-Labelled Validation Accuracy",
        "LLM-Labelled + Human Subset Validation F1 Macro",
        "LLM-Labelled + Human Subset Validation F1 Weighted",
        "LLM-Labelled + Human Subset Validation Accuracy",
    ]:
        table_df[col] = table_df[col].map(lambda x: round(float(x), 4))
    table_df["Training Time (s)"] = table_df["Training Time (s)"].map(lambda x: round(float(x), 2))
    table_df["Trainable %"] = table_df["Trainable %"].map(
        lambda x: "-" if pd.isna(x) else f"{float(x):.2f}"
    )
    table_df = table_df.sort_values("Family").reset_index(drop=True)
    table_df.to_csv(OUT_DIR / TABLE_CSV, index=False)
    return table_df


def save_markdown(table_df: pd.DataFrame) -> None:
    weak_winner = table_df.sort_values("LLM-Labelled Validation F1 Macro", ascending=False).iloc[0]
    gold_winner = table_df.sort_values(
        "LLM-Labelled + Human Subset Validation F1 Macro", ascending=False
    ).iloc[0]

    header = "| " + " | ".join(table_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    table_lines = [header, separator]
    for _, row in table_df.iterrows():
        table_lines.append("| " + " | ".join(str(row[col]) for col in table_df.columns) + " |")

    md_lines = [
        "# Model Comparison Assets for Paper",
        "",
        "Dokumen ini berisi aset ringkas yang siap ditempel ke paper atau slide.",
        "Versi ini hanya menampilkan model pada epoch 8 agar visual lebih mudah dibaca saat reporting.",
        "",
        "## Figure Captions",
        "",
        "**Figure 1. Epoch-8 comparison across the LLM-Labelled Validation and LLM-Labelled + Human Subset Validation views, using F1 Macro and F1 Weighted.**",
        "The grouped bar chart compares the four epoch-8 models on two evaluation views and shows both F1 Macro and F1 Weighted.",
        "",
        "**Figure 2. Epoch-8 training time comparison between LoRA and non-LoRA models, with LLM and human-subset accuracy annotation.**",
        "The bar chart compares training time at epoch 8 and also annotates both LLM-Labelled Validation accuracy and LLM-Labelled + Human Subset Validation accuracy for each model.",
        "",
        "**Figure 3. Training time across the four model families at epochs 3, 5, and 8.**",
        "The grouped bar chart shows how training time grows across epochs for baseline, LoRA, retrained, and retrained LoRA.",
        "",
        "**Figure 4. End-to-end funnel from raw reviews to the final model and evaluation set.**",
        "The diagram summarizes the full funnel from raw Google Play reviews, cleaning and normalization, aspect-level dataset creation, clean-subset retraining, and human-subset evaluation.",
        "",
        "## Ready-to-use Takeaway",
        "",
        f"- LLM-Labelled Validation winner at epoch 8: `{weak_winner['Family']}` with Macro F1 `{weak_winner['LLM-Labelled Validation F1 Macro']:.4f}`.",
        f"- LLM-Labelled + Human Subset Validation winner at epoch 8: `{gold_winner['Family']}` with Macro F1 `{gold_winner['LLM-Labelled + Human Subset Validation F1 Macro']:.4f}`.",
        "- Main message: the model that performs best on LLM-Labelled Validation is not the same as the model that performs best on LLM-Labelled + Human Subset Validation.",
        "- Runtime message: LoRA-based models train faster than their non-LoRA counterparts while remaining competitive in accuracy.",
        "",
        "## Table",
        "",
        *table_lines,
        "",
        "## Files",
        "",
        f"- `{EPOCH8_COMPARISON_PNG}`",
        f"- `{TRAINING_TIME_PNG}`",
        f"- `{TRAINING_TIME_ALL_EPOCHS_PNG}`",
        f"- `{PIPELINE_FUNNEL_PNG}`",
        f"- `{TABLE_CSV}`",
    ]
    (OUT_DIR / "MODEL_COMPARISON_FOR_PAPER.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = load_combined_frame()
    pipeline_numbers = load_pipeline_numbers()
    table_df = save_table(combined)
    plot_epoch8_benchmark_bar(
        table_df,
        title="Epoch 8 Model Comparison on Two Validation Views",
        out_path=OUT_DIR / EPOCH8_COMPARISON_PNG,
    )
    plot_training_time_bar(
        table_df,
        title="Epoch 8 Training Time and Accuracy: LoRA vs Non-LoRA",
        out_path=OUT_DIR / TRAINING_TIME_PNG,
    )
    plot_training_time_all_epochs(
        combined,
        title="Training Time Across Model Families and Epochs",
        out_path=OUT_DIR / TRAINING_TIME_ALL_EPOCHS_PNG,
    )
    plot_pipeline_funnel(
        pipeline_numbers,
        out_path=OUT_DIR / PIPELINE_FUNNEL_PNG,
    )
    save_markdown(table_df)
    print(f"Saved paper assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
