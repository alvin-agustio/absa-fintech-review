from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.aspect_taxonomy import ISSUE_TAXONOMY

DEFAULT_WEAK_CSV = ROOT / "droplet" / "skripsi_eval_core" / "data" / "processed" / "evaluation" / "epoch_comparison_summary.csv"
DEFAULT_GOLD_CSV = ROOT / "data" / "processed" / "diamond" / "evaluation_all_models" / "gold_evaluation_overview.csv"
LATEST_WEAK_CSV = ROOT / "droplet" / "new" / "skripsi_eval_core" / "data" / "processed" / "evaluation" / "epoch_comparison_summary.csv"
LATEST_GOLD_CSV = ROOT / "droplet" / "new" / "skripsi_eval_core" / "data" / "processed" / "diamond" / "evaluation" / "gold_evaluation_overview.csv"
V2_REPORT_JSON = ROOT / "data" / "processed" / "dataset_absa_v2_report.json"
INTERSECTION_REPORT_JSON = ROOT / "data" / "processed" / "manifests" / "stratified_50k_seed42_v2_intersection_report.json"
NOISE_REPORT_JSON = ROOT / "droplet" / "skripsi_eval_core" / "data" / "processed" / "noise" / "noise_summary.json"
OUT_DIR = ROOT / "docs" / "paper_assets"
EPOCH8_COMPARISON_PNG = "model_comparison_epoch8_benchmarks.png"
TRAINING_TIME_PNG = "model_comparison_epoch8_training_time.png"
TRAINING_TIME_ALL_EPOCHS_PNG = "model_comparison_training_time_all_epochs.png"
PIPELINE_FUNNEL_PNG = "model_building_pipeline_end_to_end.png"
TABLE_CSV = "model_comparison_epoch8_table.csv"
AGREEMENT_SCATTER_PNG = "model_agreement_llm_vs_human_best_point.png"
RETRAINING_SLOPE_PNG = "model_retraining_delta_slope.png"
TRADEOFF_BUBBLE_PNG = "model_tradeoff_time_vs_human_f1.png"
GENERALIZATION_GAP_PNG = "model_generalization_gap_best_point.png"
UNCERTAINTY_HEATMAP_PNG = "model_uncertainty_noise_heatmap.png"
ASPECT_NOISE_HEATMAP_PNG = "model_aspect_noise_heatmap.png"
EPOCH_PROGRESS_PNG = "model_progression_epoch1_15_llm_f1.png"
EPOCH_PROGRESS_NON_RETRAINED_PNG = "model_progression_epoch1_15_non_retrained.png"
EPOCH_PROGRESS_RETRAINED_PNG = "model_progression_epoch1_15_retrained.png"
BEST_POINT_BAR_PNG = "model_best_point_llm_vs_human_bar.png"
RETRAINING_DELTA_BAR_PNG = "model_retraining_delta_bar.png"
DIAGNOSIS_FLOW_PNG = "diagnosis_short_logic_end_to_end.png"
TAXONOMY_FULL_PNG = "taxonomy_complete_all_aspects.png"
TAXONOMY_FULL_CSV = "taxonomy_complete_all_aspects.csv"
RAW_REVIEW_ROWS = 505936
EXCLUDED_FAMILIES = {"qlora_trial"}

FAMILY_ORDER = [
    "baseline",
    "lora",
    "dora",
    "adalora",
    "qlora",
    "retrained",
    "retrained_lora",
    "retrained_dora",
    "retrained_adalora",
    "retrained_qlora",
]
FAMILY_LABELS = {
    "baseline": "Baseline",
    "lora": "LoRA",
    "dora": "DoRA",
    "adalora": "AdaLoRA",
    "qlora": "QLoRA",
    "retrained": "Retrained",
    "retrained_lora": "Retrained LoRA",
    "retrained_dora": "Retrained DoRA",
    "retrained_adalora": "Retrained AdaLoRA",
    "retrained_qlora": "Retrained QLoRA",
}
BENCHMARK_COLORS = {
    "LLM-Labelled Validation F1 Macro": "#B44E34",
    "LLM-Labelled + Human Subset Validation F1 Macro": "#1B7286",
}
TIME_GROUP_COLORS = {
    "full_finetune": "#B8BDC7",
    "standard_peft": "#7FA37C",
    "adaptive_quantized_peft": "#B44E34",
}
EPOCH_COLORS = {
    3: "#C9CED6",
    5: "#7FA37C",
    8: "#B44E34",
}

TIME_GROUP_LABELS = {
    "baseline": "full_finetune",
    "retrained": "full_finetune",
    "lora": "standard_peft",
    "dora": "standard_peft",
    "retrained_lora": "standard_peft",
    "retrained_dora": "standard_peft",
    "adalora": "adaptive_quantized_peft",
    "qlora": "adaptive_quantized_peft",
    "retrained_adalora": "adaptive_quantized_peft",
    "retrained_qlora": "adaptive_quantized_peft",
}


def format_minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f}m"


def build_best_point_table(df: pd.DataFrame) -> pd.DataFrame:
    best_df = (
        df.assign(family_key=df["family"].astype(str))
        .sort_values(["family_key", "weak_f1_macro", "epochs"], ascending=[True, False, True])
        .drop_duplicates(subset=["family_key"], keep="first")
        .copy()
    )
    best_df["family"] = pd.Categorical(best_df["family_key"], categories=FAMILY_ORDER, ordered=True)
    best_df = best_df.sort_values("family").reset_index(drop=True)
    best_df["Family"] = best_df["family"].map(FAMILY_LABELS)
    return best_df


def load_main_family_frame() -> pd.DataFrame:
    df = load_combined_frame().copy()
    df = df[df["family"].notna()].copy()
    df = df[~df["family"].astype(str).isin(EXCLUDED_FAMILIES)].copy()
    df["family"] = pd.Categorical(
        df["family"].astype(str),
        categories=[family for family in FAMILY_ORDER if family not in EXCLUDED_FAMILIES],
        ordered=True,
    )
    df["family_label"] = df["family"].map(FAMILY_LABELS)
    return df.sort_values(["family", "epochs"]).reset_index(drop=True)


def load_uncertainty_noise_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    uncertainty_root = ROOT / "droplet" / "new" / "skripsi_eval_core" / "data" / "processed" / "uncertainty"
    noise_root = ROOT / "droplet" / "new" / "skripsi_eval_core" / "data" / "processed" / "noise"

    uncertainty_rows = []
    for path in sorted(uncertainty_root.rglob("mc_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        family = payload.get("model_family")
        if family in EXCLUDED_FAMILIES:
            continue
        row = {
            "family": family,
            "mean_entropy": payload["mean_entropy"],
            "mean_variance": payload["mean_variance"],
            "error_rate_vs_weak": payload["error_rate_vs_weak"],
            "pred_confidence_mean": payload["pred_confidence_mean"],
        }
        for aspect, stats in payload.get("aspect_stats", {}).items():
            row[f"{aspect}_entropy"] = stats["mean_entropy"]
            row[f"{aspect}_variance"] = stats["mean_variance"]
            row[f"{aspect}_error_rate"] = stats["error_rate_vs_weak"]
        uncertainty_rows.append(row)

    noise_rows = []
    for path in sorted(noise_root.rglob("noise_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        family = payload.get("model_family")
        if family in EXCLUDED_FAMILIES:
            continue
        row = {
            "family": family,
            "noise_ratio": payload["noise_ratio"],
            "n_clean": payload["n_clean"],
            "n_noisy_candidates": payload["n_noisy_candidates"],
        }
        for aspect, stats in payload.get("full_stats", {}).get("by_aspect", {}).items():
            row[f"{aspect}_noise_ratio"] = stats["noise_ratio"]
        noise_rows.append(row)

    return pd.DataFrame(uncertainty_rows), pd.DataFrame(noise_rows)


def get_group_color(family: str) -> str:
    return TIME_GROUP_COLORS[TIME_GROUP_LABELS.get(family, "standard_peft")]


def apply_clean_axes(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Tidak menemukan file yang dibutuhkan. Cek salah satu dari: {candidates}")


def load_combined_frame() -> pd.DataFrame:
    weak = pd.read_csv(resolve_existing_path(LATEST_WEAK_CSV, DEFAULT_WEAK_CSV))
    gold = pd.read_csv(resolve_existing_path(LATEST_GOLD_CSV, DEFAULT_GOLD_CSV))

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

    gold = gold[gold["model_name"].notna()].copy()
    gold["family"] = gold["model_name"].str.replace(r"_epoch_?\d+$", "", regex=True)
    gold["epochs"] = pd.to_numeric(gold["model_name"].str.extract(r"epoch_?(\d+)")[0], errors="coerce")
    gold = gold[gold["epochs"].notna()].copy()
    gold["epochs"] = gold["epochs"].astype(int)
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

    combined = weak.merge(gold, on=["family", "epochs"], how="left")
    if combined["gold_accuracy"].isna().any():
        final_gold = (
            gold.sort_values("epochs")
            .groupby("family", as_index=False)
            .tail(1)[["family", "gold_accuracy", "gold_f1_macro", "gold_f1_weighted", "epochs"]]
            .rename(
                columns={
                    "gold_accuracy": "gold_accuracy_final",
                    "gold_f1_macro": "gold_f1_macro_final",
                    "gold_f1_weighted": "gold_f1_weighted_final",
                    "epochs": "gold_reference_epoch",
                }
            )
        )
        combined = combined.merge(final_gold, on="family", how="left")
        combined["gold_accuracy"] = combined["gold_accuracy"].fillna(combined["gold_accuracy_final"])
        combined["gold_f1_macro"] = combined["gold_f1_macro"].fillna(combined["gold_f1_macro_final"])
        combined["gold_f1_weighted"] = combined["gold_f1_weighted"].fillna(combined["gold_f1_weighted_final"])
    else:
        combined["gold_reference_epoch"] = combined["epochs"]
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
        "trained_models": len(FAMILY_ORDER),
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
        ax.set_xticklabels(family_labels, fontsize=9, rotation=18, ha="right")
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
    fig, ax = plt.subplots(figsize=(15.2, 6.8), dpi=200)

    plot_df = build_best_point_table(df)
    plot_df["family_key"] = plot_df["family"].astype(str)
    plot_df["time_group"] = plot_df["family_key"].map(TIME_GROUP_LABELS)
    family_order = [FAMILY_LABELS[item] for item in FAMILY_ORDER]
    plot_df = plot_df.set_index("Family").reindex(family_order).reset_index()
    plot_df["training_time_minutes"] = plot_df["training_time_seconds"] / 60.0
    colors = [
        TIME_GROUP_COLORS[plot_df.loc[idx, "time_group"]]
        for idx, _ in enumerate(plot_df["Family"])
    ]

    bars = ax.bar(
        plot_df["Family"],
        plot_df["training_time_minutes"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.58,
    )

    max_value = float(plot_df["training_time_minutes"].max())
    offset = max_value * 0.035
    for bar, value, weak_f1, gold_f1 in zip(
        bars,
        plot_df["training_time_seconds"],
        plot_df["weak_f1_macro"],
        plot_df["gold_f1_macro"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) / 60.0 + offset,
            f"{format_minutes(float(value))}\nWeak {float(weak_f1):.3f}\nGold {float(gold_f1):.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#30333A",
        )

    ax.set_ylabel("Training Time (minutes)", fontsize=10)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylim(0, max_value * 1.18)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["full_finetune"]),
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["standard_peft"]),
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["adaptive_quantized_peft"]),
    ]
    ax.legend(
        handles,
        ["Full Finetune", "Standard PEFT", "Adaptive / Quantized PEFT"],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.subplots_adjust(bottom=0.18, top=0.86)
    fig.text(
        0.5,
        0.04,
        "Weak = best weak-label Macro-F1 point per family; Gold = latest available gold Macro-F1",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#50544e",
    )

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
    ax.set_xticklabels(family_labels, fontsize=9, rotation=18, ha="right")
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
        ("Pelatihan model", f"{numbers['trained_models']} keluarga model", "Keluarga model final yang dianalisis pada bundle eksperimen terbaru"),
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


def plot_agreement_scatter(best_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10.2, 7.2), dpi=200)
    plot_df = best_df.sort_values("gold_f1_macro", ascending=True).reset_index(drop=True)
    y_pos = list(range(len(plot_df)))
    bar_h = 0.36

    ax.barh(
        [y - bar_h / 2 for y in y_pos],
        plot_df["weak_f1_macro"],
        height=bar_h,
        color="#B44E34",
        edgecolor="white",
        linewidth=0.8,
        label="Best LLM-Labelled Macro-F1",
    )
    ax.barh(
        [y + bar_h / 2 for y in y_pos],
        plot_df["gold_f1_macro"],
        height=bar_h,
        color="#1B7286",
        edgecolor="white",
        linewidth=0.8,
        label="Human-Subset Macro-F1",
    )

    for idx, row in plot_df.iterrows():
        ax.text(float(row["weak_f1_macro"]) + 0.003, idx - bar_h / 2, f'{float(row["weak_f1_macro"]):.3f}', va="center", fontsize=8, color="#30333A")
        ax.text(float(row["gold_f1_macro"]) + 0.003, idx + bar_h / 2, f'{float(row["gold_f1_macro"]):.3f}', va="center", fontsize=8, color="#30333A")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Family"], fontsize=9)
    ax.set_xlabel("Macro-F1", fontsize=10)
    ax.set_title("Best LLM-Labelled vs Human-Subset Macro-F1 by Family", fontsize=13, weight="bold")
    apply_clean_axes(ax)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#B44E34", label="Best LLM-Labelled Macro-F1"),
        plt.Rectangle((0, 0), 1, 1, color="#1B7286", label="Human-Subset Macro-F1"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")


def plot_best_point_bar(best_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(13.2, 6.4), dpi=200)

    plot_df = best_df.copy()
    plot_df["best_epoch_label"] = "E" + plot_df["epochs"].astype(int).astype(str)
    x = list(range(len(plot_df)))
    width = 0.36

    weak_bars = ax.bar(
        [pos - width / 2 for pos in x],
        plot_df["weak_f1_macro"],
        width=width,
        color="#B44E34",
        edgecolor="white",
        linewidth=0.8,
        label="LLM-Labelled Macro-F1",
    )
    human_bars = ax.bar(
        [pos + width / 2 for pos in x],
        plot_df["gold_f1_macro"],
        width=width,
        color="#1B7286",
        edgecolor="white",
        linewidth=0.8,
        label="Human-Subset Macro-F1",
    )

    for bar, epoch_label in zip(weak_bars, plot_df["best_epoch_label"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            epoch_label,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#30333A",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Family"], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Macro-F1", fontsize=10)
    ax.set_title("Best LLM-Labelled Point Versus Human-Subset Outcome", fontsize=13, weight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    apply_clean_axes(ax)
    fig.text(
        0.5,
        0.02,
        "Labels above the LLM bars show the epoch where each family reached its best LLM-labelled Macro-F1.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#50544e",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")


def plot_retraining_slope(best_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), dpi=200)
    pairs = [
        ("baseline", "retrained"),
        ("lora", "retrained_lora"),
        ("dora", "retrained_dora"),
        ("adalora", "retrained_adalora"),
        ("qlora", "retrained_qlora"),
    ]

    for ax, metric, title in [
        (axes[0], "weak_f1_macro", "LLM-Labelled Macro-F1"),
        (axes[1], "gold_f1_macro", "Human-Subset Macro-F1"),
    ]:
        x = list(range(len(pairs)))
        width = 0.34
        base_values = []
        retrained_values = []
        labels = []
        for base_family, retrained_family in pairs:
            base_row = best_df[best_df["family"].astype(str) == base_family].iloc[0]
            retrained_row = best_df[best_df["family"].astype(str) == retrained_family].iloc[0]
            base_values.append(float(base_row[metric]))
            retrained_values.append(float(retrained_row[metric]))
            labels.append(FAMILY_LABELS[base_family])

        base_bars = ax.bar(
            [pos - width / 2 for pos in x],
            base_values,
            width=width,
            color="#B8BDC7",
            edgecolor="white",
            linewidth=0.8,
            label="Before Retraining",
        )
        retrained_bars = ax.bar(
            [pos + width / 2 for pos in x],
            retrained_values,
            width=width,
            color="#B44E34",
            edgecolor="white",
            linewidth=0.8,
            label="After Retraining",
        )

        for bars in [base_bars, retrained_bars]:
            for bar in bars:
                value = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5, color="#30333A")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=18, ha="right")
        ax.set_title(title, fontsize=11, weight="bold")
        apply_clean_axes(ax)

    axes[0].set_ylabel("Score", fontsize=10)
    fig.suptitle("Effect of Uncertainty-Aware Retraining Across Model Families", fontsize=13, weight="bold", y=1.02)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#B8BDC7", label="Before Retraining"),
        plt.Rectangle((0, 0), 1, 1, color="#B44E34", label="After Retraining"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, bbox_inches="tight")


def plot_retraining_delta_bar(best_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), dpi=200)
    pairs = [
        ("baseline", "retrained"),
        ("lora", "retrained_lora"),
        ("dora", "retrained_dora"),
        ("adalora", "retrained_adalora"),
        ("qlora", "retrained_qlora"),
    ]

    for ax, metric, title in [
        (axes[0], "weak_f1_macro", "Delta on LLM-Labelled Macro-F1"),
        (axes[1], "gold_f1_macro", "Delta on Human-Subset Macro-F1"),
    ]:
        labels = []
        deltas = []
        colors = []
        for base_family, retrained_family in pairs:
            base_row = best_df[best_df["family"].astype(str) == base_family].iloc[0]
            retrained_row = best_df[best_df["family"].astype(str) == retrained_family].iloc[0]
            labels.append(FAMILY_LABELS[base_family].replace("Retrained ", ""))
            delta = float(retrained_row[metric] - base_row[metric])
            deltas.append(delta)
            colors.append("#7FA37C" if delta >= 0 else "#B44E34")

        bars = ax.bar(labels, deltas, color=colors, edgecolor="white", linewidth=0.8)
        for bar, delta in zip(bars, deltas):
            va = "bottom" if delta >= 0 else "top"
            pad = 0.004 if delta >= 0 else -0.004
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                delta + pad,
                f"{delta:+.3f}",
                ha="center",
                va=va,
                fontsize=8,
                color="#30333A",
            )

        ax.axhline(0, color="#8c8f95", linewidth=1.0)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.tick_params(axis="x", rotation=18)
        apply_clean_axes(ax)

    axes[0].set_ylabel("Retraining Delta", fontsize=10)
    fig.suptitle("Effect of Retraining as a Clean Before-vs-After Delta", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_tradeoff_bubble(best_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.8), dpi=200, sharex=True)

    plot_df = best_df.copy()
    plot_df["training_time_minutes"] = plot_df["training_time_seconds"] / 60.0
    size_series = plot_df["trainable_pct"].fillna(1.0).astype(float)
    sizes = 280 + size_series * 420

    panel_specs = [
        (axes[0], "weak_f1_macro", "LLM-Labelled Macro-F1"),
        (axes[1], "gold_f1_macro", "Human-Subset Macro-F1"),
    ]

    for ax, metric, panel_title in panel_specs:
        for (_, row), size in zip(plot_df.iterrows(), sizes):
            family = str(row["family"])
            ax.scatter(
                row["training_time_minutes"],
                row[metric],
                s=size,
                color=get_group_color(family),
                alpha=0.88,
                edgecolor="white",
                linewidth=1.0,
            )
            ax.text(
                row["training_time_minutes"] + 0.9,
                row[metric] + 0.002,
                row["Family"],
                fontsize=8.1,
                color="#30333A",
            )
        ax.set_title(panel_title, fontsize=11, weight="bold")
        ax.set_xlabel("Training Time at Best LLM Point (minutes)", fontsize=10)
        apply_clean_axes(ax, axis="both")

    axes[0].set_ylabel("Macro-F1", fontsize=10)
    fig.suptitle("Practical Trade-off: Training Cost vs Validation Quality", fontsize=13, weight="bold", y=0.98)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["full_finetune"], label="Full Finetune"),
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["standard_peft"], label="Standard PEFT"),
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["adaptive_quantized_peft"], label="Adaptive / Quantized PEFT"),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3)
    fig.text(
        0.5,
        0.02,
        "Bubble size reflects trainable parameter share. Left: LLM-labelled quality. Right: human-subset quality.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#50544e",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    fig.savefig(out_path, bbox_inches="tight")


def plot_generalization_gap(best_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=200)

    plot_df = best_df.copy()
    plot_df["gap"] = plot_df["weak_f1_macro"] - plot_df["gold_f1_macro"]
    plot_df = plot_df.sort_values("gap", ascending=False).reset_index(drop=True)
    colors = [get_group_color(str(family)) for family in plot_df["family"]]
    bars = ax.bar(plot_df["Family"], plot_df["gap"], color=colors, edgecolor="white", linewidth=0.8)
    for bar, value in zip(bars, plot_df["gap"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#30333A")

    ax.set_ylabel("LLM-to-Human Macro-F1 Gap", fontsize=10)
    ax.set_title("Generalization Gap from LLM-Labelled Validation to Human Subset", fontsize=13, weight="bold")
    ax.tick_params(axis="x", rotation=18)
    apply_clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_uncertainty_heatmap(best_df: pd.DataFrame, uncertainty_df: pd.DataFrame, noise_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    merged = best_df.merge(uncertainty_df, on="family", how="left").merge(noise_df, on="family", how="left")
    plot_df = merged[merged["mean_entropy"].notna()].copy()
    plot_df["family"] = pd.Categorical(plot_df["family"], categories=[family for family in FAMILY_ORDER if family not in EXCLUDED_FAMILIES], ordered=True)
    plot_df = plot_df.sort_values("family")

    columns = ["mean_entropy", "mean_variance", "error_rate_vs_weak", "noise_ratio", "gold_f1_macro"]
    labels = ["Entropy", "Variance", "Mismatch", "Noise Ratio", "Human F1"]
    norm_df = plot_df[columns].copy()
    for col in columns:
        col_min = norm_df[col].min()
        col_max = norm_df[col].max()
        norm_df[col] = 0.5 if col_max == col_min else (norm_df[col] - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=200)
    im = ax.imshow(norm_df.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Family"], fontsize=9)
    ax.set_title("Uncertainty, Noise, and Human-Subset Outcome Across Families", fontsize=13, weight="bold")
    for i in range(len(plot_df)):
        for j, col in enumerate(columns):
            ax.text(j, i, f"{plot_df.iloc[i][col]:.3f}", ha="center", va="center", fontsize=7.5, color="#1f211e")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Within-column normalized intensity")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_aspect_noise_heatmap(noise_df: pd.DataFrame, out_path: Path) -> None:
    plt.close("all")
    plot_df = noise_df.copy()
    plot_df = plot_df[~plot_df["family"].isin(EXCLUDED_FAMILIES)].copy()
    plot_df["family"] = pd.Categorical(plot_df["family"], categories=[family for family in FAMILY_ORDER if family not in EXCLUDED_FAMILIES], ordered=True)
    plot_df = plot_df.sort_values("family")
    columns = ["service_noise_ratio", "risk_noise_ratio", "trust_noise_ratio"]
    labels = ["Service", "Risk", "Trust"]

    fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=200)
    im = ax.imshow(plot_df[columns].values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["family"].map(FAMILY_LABELS), fontsize=9)
    ax.set_title("Aspect-Level Noise Ratio After MC-Dropout Filtering", fontsize=13, weight="bold")
    for i in range(len(plot_df)):
        for j, col in enumerate(columns):
            ax.text(j, i, f"{plot_df.iloc[i][col]:.3f}", ha="center", va="center", fontsize=7.5, color="#1f211e")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Noise ratio")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_epoch_progress_lines(df: pd.DataFrame, out_path: Path, families: list[str], title: str, footnote: str) -> None:
    plt.close("all")
    n = len(families)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 2.9 * nrows), dpi=200, sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, family in zip(axes, families):
        subset = df[df["family"].astype(str) == family].sort_values("epochs")
        ax.bar(
            subset["epochs"],
            subset["weak_f1_macro"],
            color=get_group_color(family),
            edgecolor="white",
            linewidth=0.6,
            width=0.8,
        )
        best_epoch = subset.loc[subset["weak_f1_macro"].idxmax(), "epochs"]
        best_val = subset["weak_f1_macro"].max()
        ax.set_title(FAMILY_LABELS[family], fontsize=10, weight="bold")
        ax.text(0.98, 0.92, f"Best e{int(best_epoch)}\n{float(best_val):.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#30333A")
        ax.set_xticks([1, 5, 10, 15])
        apply_clean_axes(ax)

    for ax in axes[len(families):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, weight="bold", y=0.995)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["full_finetune"], label="Full Finetune"),
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["standard_peft"], label="Standard PEFT"),
        plt.Rectangle((0, 0), 1, 1, color=TIME_GROUP_COLORS["adaptive_quantized_peft"], label="Adaptive / Quantized PEFT"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
    )
    fig.supxlabel("Epoch", fontsize=10)
    fig.supylabel("Macro-F1", fontsize=10)
    fig.text(
        0.5,
        0.02,
        footnote,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#50544e",
    )
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")


def _add_round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#1f211e",
    fontsize: float = 10,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f211e",
        weight=weight,
        wrap=True,
    )


def _add_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#50544e",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def plot_diagnosis_logic_flow(out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(11.2, 12.8), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "Alur Logika Diagnosis Singkat End-to-End",
        ha="center",
        va="top",
        fontsize=16,
        weight="bold",
        color="#1f211e",
    )
    ax.text(
        0.5,
        0.947,
        "Diagnosis singkat dibentuk dari kombinasi prediksi model ABSA dan aturan eksplisit yang bisa diaudit. Baca alurnya dari atas ke bawah.",
        ha="center",
        va="top",
        fontsize=10,
        color="#50544e",
    )

    stage_colors = {
        "input": "#F7F1E5",
        "prep": "#E7F1EF",
        "model": "#EAF1F8",
        "rule": "#F6E9E2",
        "output": "#ECE8F3",
    }
    steps = [
        ("Input", "1. Ambil review mentah\nDari Google Play atau cache live fetch sebelumnya", "input"),
        ("Preprocessing", "2. Bersihkan review\nFilter tanggal, buang duplikasi, bersihkan teks, dan sisakan review yang valid", "prep"),
        ("Model ABSA", "3. Prediksi sentimen 3 aspek\nModel memprediksi risk, trust, dan service untuk setiap review", "model"),
        ("Rule", "4. Cek aspect presence\nAturan keyword memeriksa apakah aspek itu benar-benar disebut", "rule"),
        ("Rule", "5. Fokus ke aspek yang hadir\nHanya aspek yang benar-benar hadir dipakai untuk diagnosis dan bukti", "rule"),
        ("Rule", "6. Cocokkan ke taxonomy issue\nReview dibandingkan dengan bucket issue milik aspek yang sama", "rule"),
        ("Rule", "7. Agregasi dan pilih bukti\nHitung issue dominan, keyword dominan, app dominan, lalu pilih review paling representatif", "rule"),
        ("Output", "8. Susun diagnosis singkat\nRule summary membentuk gambaran umum, sinyal utama, makna akhir, dan kartu dashboard", "output"),
    ]

    x_box = 0.22
    w_box = 0.70
    h_box = 0.082
    y_top = 0.84
    gap = 0.03
    phase_x = 0.05
    phase_w = 0.14

    for idx, (phase, text, kind) in enumerate(steps):
        y = y_top - idx * (h_box + gap)
        _add_round_box(ax, phase_x, y, phase_w, h_box, phase, stage_colors[kind], fontsize=10, weight="bold")
        _add_round_box(ax, x_box, y, w_box, h_box, text, stage_colors[kind], fontsize=10 if idx != 2 else 10.4, weight="bold" if idx == 2 else "normal")
        if idx < len(steps) - 1:
            _add_arrow(ax, x_box + w_box / 2, y, x_box + w_box / 2, y - gap + 0.004)

    ax.text(
        0.5,
        0.035,
        "Kesimpulan penting: issue hierarchy dan diagnosis singkat bukan keluaran langsung model. Keduanya dibentuk sesudah inferensi ABSA dengan aturan yang eksplisit dan dapat diaudit.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#50544e",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def plot_taxonomy_complete(out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(18.0, 13.6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.975,
        "Taxonomy Rule-Based Lengkap untuk Diagnosis Issue per Aspek",
        ha="center",
        va="top",
        fontsize=16,
        weight="bold",
        color="#1f211e",
    )
    ax.text(
        0.5,
        0.952,
        "Setiap aspek memiliki bucket issue sendiri. Satu review hanya dibandingkan dengan bucket pada aspek yang sama, lalu rule dengan kecocokan terkuat yang dipilih.",
        ha="center",
        va="top",
        fontsize=10,
        color="#50544e",
    )

    columns = [
        ("risk", 0.035, 0.295, "#F6E3DB", "Risk"),
        ("trust", 0.3525, 0.295, "#DDEDF2", "Trust"),
        ("service", 0.67, 0.295, "#E3F0E7", "Service"),
    ]

    keyword_examples = {
        rule.label: ", ".join(rule.keywords[:3]) + (", ..." if len(rule.keywords) > 3 else "")
        for aspect_rules in ISSUE_TAXONOMY.values()
        for rule in aspect_rules
    }
    descriptions = {
        "Limit, approval, dan pencairan": "Masalah limit, persetujuan, penolakan, atau dana belum cair.",
        "Bunga, biaya, dan denda": "Keluhan bunga, biaya admin, cicilan, atau denda.",
        "Penagihan dan debt collector": "Penagihan agresif, ancaman, atau kontak berulang.",
        "Blokir setelah pembayaran": "Akun tetap bermasalah walau pengguna merasa sudah membayar.",
        "Keamanan data pribadi": "Privasi, akses kontak, atau data pribadi bocor/disalahgunakan.",
        "Transparansi dan kejelasan": "Alasan, status, atau proses terasa tidak jelas.",
        "Legalitas, OJK, dan reputasi": "Legalitas, izin, reputasi, dan kepercayaan formal.",
        "Penipuan dan fraud": "Kesan penipuan, scam, modus, atau perilaku yang terasa merugikan.",
        "Peretasan dan keamanan akun": "Akun diretas, dibobol, atau sistem terasa tidak aman.",
        "Akun, fairness, dan suspend": "Suspend, blokir akun, atau keputusan yang terasa tidak adil.",
        "Privasi dan penyalahgunaan data": "Penyalahgunaan data, akses kontak, dan kebocoran privasi.",
        "Komunikasi dan kepastian proses": "Kepastian kabar, notifikasi, dan kualitas komunikasi.",
        "Bug, error, dan stabilitas": "Error teknis, crash, loading, dan fungsi inti terganggu.",
        "Pendaftaran, login, dan verifikasi": "Masalah akses aplikasi, daftar, login, OTP, aktivasi, atau verifikasi.",
        "CS dan respon admin": "Respon customer service, bantuan, dan tindak lanjut admin.",
        "Fitur, pencarian, dan katalog": "Masalah fitur, pencarian, katalog, atau alur konfirmasi/penerimaan.",
        "Proses dan pencairan": "Proses transaksi, pending, review, transfer, atau pencairan.",
        "Kemudahan penggunaan dan UX": "Aplikasi terasa ribet, sulit dipakai, atau membingungkan.",
        "Performa dan stabilitas aplikasi": "Lemot, lag, berat, atau gangguan performa aplikasi.",
        "Update dan gangguan aplikasi": "Gangguan setelah update, versi baru, atau downtime.",
    }

    top_y = 0.865
    gap_y = 0.014
    for aspect, x, w, color, title in columns:
        rules = ISSUE_TAXONOMY[aspect]
        title_h = 0.06
        _add_round_box(ax, x, top_y, w, title_h, f"{title}\n{len(rules)} issue buckets", color, fontsize=11.2, weight="bold")

        available_h = 0.77
        card_h = (available_h - gap_y * (len(rules) - 1)) / len(rules)
        y = top_y - 0.025 - card_h
        for rule in rules:
            text = (
                f"{rule.label}\n"
                f"Prioritas {rule.priority} | {descriptions[rule.label]}\n"
                f"Contoh keyword: {keyword_examples[rule.label]}"
            )
            _add_round_box(ax, x, y, w, card_h, text, "#FCFBF7", edgecolor=color, fontsize=8.7, weight="bold" if rule.priority >= 5 else "normal")
            y -= card_h + gap_y

    ax.text(
        0.5,
        0.025,
        "Aturan keputusan: review dinormalisasi, dicocokkan hanya dengan bucket pada aspek yang dipilih, lalu label dengan signature keyword berbobot yang paling kuat dipilih. Signature ini mempertimbangkan bobot keyword, spesifisitas frasa, dan prioritas rule. Jika tidak ada bucket yang cocok, sistem memberi label fallback 'Belum cukup spesifik'.",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#50544e",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def save_taxonomy_csv(out_path: Path) -> pd.DataFrame:
    descriptions = {
        "Limit, approval, dan pencairan": "Masalah limit, persetujuan, penolakan, atau dana belum cair.",
        "Bunga, biaya, dan denda": "Keluhan bunga, biaya admin, cicilan, atau denda.",
        "Penagihan dan debt collector": "Penagihan agresif, ancaman, atau kontak berulang.",
        "Blokir setelah pembayaran": "Akun tetap bermasalah walau pengguna merasa sudah membayar.",
        "Keamanan data pribadi": "Privasi, akses kontak, atau data pribadi bocor/disalahgunakan.",
        "Transparansi dan kejelasan": "Alasan, status, atau proses terasa tidak jelas.",
        "Legalitas, OJK, dan reputasi": "Legalitas, izin, reputasi, dan kepercayaan formal.",
        "Penipuan dan fraud": "Kesan penipuan, scam, modus, atau perilaku yang terasa merugikan.",
        "Peretasan dan keamanan akun": "Akun diretas, dibobol, atau sistem terasa tidak aman.",
        "Akun, fairness, dan suspend": "Suspend, blokir akun, atau keputusan yang terasa tidak adil.",
        "Privasi dan penyalahgunaan data": "Penyalahgunaan data, akses kontak, dan kebocoran privasi.",
        "Komunikasi dan kepastian proses": "Kepastian kabar, notifikasi, dan kualitas komunikasi.",
        "Bug, error, dan stabilitas": "Error teknis, crash, loading, dan fungsi inti terganggu.",
        "Pendaftaran, login, dan verifikasi": "Masalah akses aplikasi, daftar, login, OTP, aktivasi, atau verifikasi.",
        "CS dan respon admin": "Respon customer service, bantuan, dan tindak lanjut admin.",
        "Fitur, pencarian, dan katalog": "Masalah fitur, pencarian, katalog, atau alur konfirmasi/penerimaan.",
        "Proses dan pencairan": "Proses transaksi, pending, review, transfer, atau pencairan.",
        "Kemudahan penggunaan dan UX": "Aplikasi terasa ribet, sulit dipakai, atau membingungkan.",
        "Performa dan stabilitas aplikasi": "Lemot, lag, berat, atau gangguan performa aplikasi.",
        "Update dan gangguan aplikasi": "Gangguan setelah update, versi baru, atau downtime.",
    }

    rows: list[dict[str, object]] = []
    for aspect, rules in ISSUE_TAXONOMY.items():
        for order_idx, rule in enumerate(rules, start=1):
            rows.append(
                {
                    "aspect": aspect,
                    "aspect_display": aspect.title(),
                    "bucket_order_within_aspect": order_idx,
                    "issue_label": rule.label,
                    "priority": rule.priority,
                    "description": descriptions[rule.label],
                    "keyword_count": len(rule.keywords),
                    "keywords": " | ".join(rule.keywords),
                    "example_keywords": " | ".join(rule.keywords[:5]),
                    "fallback_label_if_no_match": "Belum cukup spesifik",
                    "decision_rule_note": "Label dipilih dari signature keyword berbobot terkuat dalam aspek yang sama.",
                }
            )

    taxonomy_df = pd.DataFrame(rows)
    taxonomy_df.to_csv(out_path, index=False, encoding="utf-8")
    return taxonomy_df


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
    table_df = table_df[table_df["Family"].notna()].copy()
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
        "Versi ini menggabungkan perbandingan epoch 8 untuk metrik validasi dan best-point comparison untuk grafik training time.",
        "",
        "## Figure Captions",
        "",
        "**Figure 1. Epoch-8 comparison across the LLM-Labelled Validation and LLM-Labelled + Human Subset Validation views, using F1 Macro and F1 Weighted.**",
        "The grouped bar chart compares the epoch-8 model families on two evaluation views and shows both F1 Macro and F1 Weighted.",
        "",
        "**Figure 2. Training time at the best weak-label validation point across model families, annotated with the latest available gold Macro-F1.**",
        "The bar chart compares training time at the best weak-label Macro-F1 point for each family and annotates each family with that weak-label Macro-F1 plus the latest available gold Macro-F1 from the final evaluated run.",
        "",
        "**Figure 3. Training time across model families at epochs 3, 5, and 8.**",
        "The grouped bar chart shows how training time grows across the available model families at epochs 3, 5, and 8.",
        "",
        "**Figure 4. End-to-end funnel from raw reviews to the final model and evaluation set.**",
        "The diagram summarizes the full funnel from raw Google Play reviews, cleaning and normalization, aspect-level dataset creation, clean-subset retraining, and human-subset evaluation.",
        "",
        "**Figure 5. Agreement between the LLM-labelled validation set and the manual human subset.**",
        "The scatter plot contrasts the best LLM-labelled Macro-F1 point for each family against the latest available human-subset Macro-F1, highlighting models that generalize well and those that overfit the weak labels.",
        "",
        "**Figure 5a. Best-point grouped bar chart across the LLM-labelled and human-subset views.**",
        "The grouped bar chart compares each family's best LLM-labelled Macro-F1 against its latest available human-subset Macro-F1, with the best LLM epoch shown above each bar.",
        "",
        "**Figure 6. Before-versus-after retraining slope chart.**",
        "The slope chart compares each base family with its retrained counterpart on both the LLM-labelled validation set and the manual human subset.",
        "",
        "**Figure 6a. Retraining delta bar chart.**",
        "The delta bar chart shows how much each family changes after retraining on the LLM-labelled validation set and on the manual human subset.",
        "",
        "**Figure 7. Training-cost versus validation-quality trade-off.**",
        "The two-panel bubble chart compares training time at each family's best LLM-labelled point against both LLM-labelled Macro-F1 and human-subset Macro-F1, while bubble size reflects trainable parameter share.",
        "",
        "**Figure 8. Generalization gap from LLM-labelled to human-subset validation.**",
        "The bar chart shows how far each family drops when moving from the weak-label setting to the manual human-subset setting.",
        "",
        "**Figure 9. Uncertainty, noise, and outcome heatmap.**",
        "The heatmap combines MC-dropout entropy, variance, mismatch rate, noise ratio, and final human-subset Macro-F1 for the families that have uncertainty artifacts.",
        "",
        "**Figure 10. Aspect-level noise ratio heatmap.**",
        "The heatmap breaks the detected noise ratio into the service, risk, and trust aspects for the families with noise-detection artifacts.",
        "",
        "**Figure 11. End-to-end short-diagnosis logic.**",
        "The flow diagram explains how the dashboard moves from raw review text to ABSA sentiment prediction, aspect-presence filtering, issue taxonomy assignment, evidence ranking, and final short diagnosis cards.",
        "",
        "**Figure 12. Complete rule-based taxonomy across all aspects.**",
        "The taxonomy map lists every issue bucket for risk, trust, and service, together with its intent and example keywords.",
        "",
        "**Appendix CSV. Complete taxonomy table across all aspects.**",
        "The CSV version lists every aspect, bucket, priority, description, and keyword set in a tabular form for audit and appendix use.",
        "",
        "## Ready-to-use Takeaway",
        "",
        f"- LLM-Labelled Validation winner at epoch 8: `{weak_winner['Family']}` with Macro F1 `{weak_winner['LLM-Labelled Validation F1 Macro']:.4f}`.",
        f"- LLM-Labelled + Human Subset Validation winner at epoch 8: `{gold_winner['Family']}` with Macro F1 `{gold_winner['LLM-Labelled + Human Subset Validation F1 Macro']:.4f}`.",
        "- Main message: the model that performs best on LLM-Labelled Validation is not the same as the model that performs best on LLM-Labelled + Human Subset Validation.",
        "- Runtime message: standard PEFT families remain the fastest, while QLoRA-based families are the slowest in the latest run set.",
        "- Practical message: the most useful operational comparison is the trade-off between training time and human-subset quality, not the raw epoch-by-epoch progression alone.",
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
        f"- `{AGREEMENT_SCATTER_PNG}`",
        f"- `{BEST_POINT_BAR_PNG}`",
        f"- `{RETRAINING_SLOPE_PNG}`",
        f"- `{RETRAINING_DELTA_BAR_PNG}`",
        f"- `{TRADEOFF_BUBBLE_PNG}`",
        f"- `{GENERALIZATION_GAP_PNG}`",
        f"- `{UNCERTAINTY_HEATMAP_PNG}`",
        f"- `{ASPECT_NOISE_HEATMAP_PNG}`",
        f"- `{DIAGNOSIS_FLOW_PNG}`",
        f"- `{TAXONOMY_FULL_PNG}`",
        f"- `{TAXONOMY_FULL_CSV}`",
        f"- `{TABLE_CSV}`",
    ]
    (OUT_DIR / "MODEL_COMPARISON_FOR_PAPER.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = load_main_family_frame()
    pipeline_numbers = load_pipeline_numbers()
    uncertainty_df, noise_df = load_uncertainty_noise_summary()
    best_df = build_best_point_table(combined)
    table_df = save_table(combined)
    plot_epoch8_benchmark_bar(
        table_df,
        title="Epoch 8 Model Comparison on Two Validation Views",
        out_path=OUT_DIR / EPOCH8_COMPARISON_PNG,
    )
    plot_training_time_bar(
        combined,
        title="Training Time at Best Weak-Label Validation Point",
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
    plot_agreement_scatter(best_df, OUT_DIR / AGREEMENT_SCATTER_PNG)
    plot_best_point_bar(best_df, OUT_DIR / BEST_POINT_BAR_PNG)
    plot_retraining_slope(best_df, OUT_DIR / RETRAINING_SLOPE_PNG)
    plot_retraining_delta_bar(best_df, OUT_DIR / RETRAINING_DELTA_BAR_PNG)
    plot_tradeoff_bubble(best_df, OUT_DIR / TRADEOFF_BUBBLE_PNG)
    plot_generalization_gap(best_df, OUT_DIR / GENERALIZATION_GAP_PNG)
    plot_uncertainty_heatmap(best_df, uncertainty_df, noise_df, OUT_DIR / UNCERTAINTY_HEATMAP_PNG)
    plot_aspect_noise_heatmap(noise_df, OUT_DIR / ASPECT_NOISE_HEATMAP_PNG)
    plot_diagnosis_logic_flow(OUT_DIR / DIAGNOSIS_FLOW_PNG)
    plot_taxonomy_complete(OUT_DIR / TAXONOMY_FULL_PNG)
    save_taxonomy_csv(OUT_DIR / TAXONOMY_FULL_CSV)
    plot_epoch_progress_lines(
        combined,
        OUT_DIR / EPOCH_PROGRESS_NON_RETRAINED_PNG,
        families=["baseline", "lora", "dora", "adalora", "qlora"],
        title="Epoch-by-Epoch Progression for Non-Retrained Families",
        footnote="Each panel shows one non-retrained family as a bar chart across epochs 1-15.",
    )
    plot_epoch_progress_lines(
        combined,
        OUT_DIR / EPOCH_PROGRESS_RETRAINED_PNG,
        families=["retrained", "retrained_lora", "retrained_dora", "retrained_adalora", "retrained_qlora"],
        title="Epoch-by-Epoch Progression for Retrained Families",
        footnote="Each panel shows one retrained family as a bar chart across epochs 1-15.",
    )
    save_markdown(table_df)
    print(f"Saved paper assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
