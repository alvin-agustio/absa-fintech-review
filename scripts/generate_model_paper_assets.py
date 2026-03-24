from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEAK_CSV = ROOT / "droplet" / "skripsi_eval_core" / "data" / "processed" / "evaluation" / "epoch_comparison_summary.csv"
GOLD_CSV = ROOT / "data" / "processed" / "diamond" / "evaluation_all_models" / "gold_evaluation_overview.csv"
OUT_DIR = ROOT / "docs" / "paper_assets"

FAMILY_ORDER = ["baseline", "lora", "retrained", "retrained_lora"]
FAMILY_LABELS = {
    "baseline": "Baseline",
    "lora": "LoRA",
    "retrained": "Retrained",
    "retrained_lora": "Retrained LoRA",
}
EPOCHS = [3, 5, 8]
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
        }
    )
    weak = weak[["family", "epochs", "weak_accuracy", "weak_f1_macro", "training_time_seconds", "trainable_pct"]].copy()

    gold["family"] = gold["model_name"].str.replace(r"_epoch\d+$", "", regex=True)
    gold["epochs"] = gold["model_name"].str.extract(r"epoch(\d+)").astype(int)
    gold = gold.rename(
        columns={
            "sentiment_accuracy_present": "gold_accuracy",
            "sentiment_f1_macro_present": "gold_f1_macro",
        }
    )
    gold = gold[["family", "epochs", "gold_accuracy", "gold_f1_macro", "model_type"]].copy()

    combined = weak.merge(gold, on=["family", "epochs"], how="inner")
    combined["family"] = pd.Categorical(combined["family"], categories=FAMILY_ORDER, ordered=True)
    combined = combined.sort_values(["family", "epochs"]).reset_index(drop=True)
    combined["family_label"] = combined["family"].map(FAMILY_LABELS)
    combined["model_name"] = combined["family"].astype(str) + "_epoch" + combined["epochs"].astype(str)
    return combined


def plot_grouped_bar(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)

    family_labels = [FAMILY_LABELS[item] for item in FAMILY_ORDER]
    x = list(range(len(FAMILY_ORDER)))
    width = 0.22
    offsets = {3: -width, 5: 0.0, 8: width}

    for epoch in EPOCHS:
        subset = df[df["epochs"] == epoch].set_index("family").reindex(FAMILY_ORDER)
        values = subset[metric].tolist()
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
                value + 0.004,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#30333A",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(family_labels, fontsize=10)
    ax.set_ylim(0.58, 0.91 if "weak" in metric else 0.84)
    ax.set_ylabel("Macro F1", fontsize=10)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def save_table(df: pd.DataFrame) -> pd.DataFrame:
    table_df = df[
        [
            "family_label",
            "epochs",
            "weak_f1_macro",
            "weak_accuracy",
            "gold_f1_macro",
            "gold_accuracy",
            "training_time_seconds",
            "trainable_pct",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "family_label": "Family",
            "epochs": "Epoch",
            "weak_f1_macro": "Weak F1 Macro",
            "weak_accuracy": "Weak Accuracy",
            "gold_f1_macro": "Gold F1 Macro",
            "gold_accuracy": "Gold Accuracy",
            "training_time_seconds": "Training Time (s)",
            "trainable_pct": "Trainable %",
        }
    )
    for col in ["Weak F1 Macro", "Weak Accuracy", "Gold F1 Macro", "Gold Accuracy"]:
        table_df[col] = table_df[col].map(lambda x: round(float(x), 4))
    table_df["Training Time (s)"] = table_df["Training Time (s)"].map(lambda x: round(float(x), 2))
    table_df["Trainable %"] = table_df["Trainable %"].map(
        lambda x: "-" if pd.isna(x) else f"{float(x):.2f}"
    )
    table_df.to_csv(OUT_DIR / "model_comparison_table.csv", index=False)
    return table_df


def save_markdown(table_df: pd.DataFrame) -> None:
    weak_winner = table_df.sort_values("Weak F1 Macro", ascending=False).iloc[0]
    gold_winner = table_df.sort_values("Gold F1 Macro", ascending=False).iloc[0]

    header = "| " + " | ".join(table_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    table_lines = [header, separator]
    for _, row in table_df.iterrows():
        table_lines.append("| " + " | ".join(str(row[col]) for col in table_df.columns) + " |")

    md_lines = [
        "# Model Comparison Assets for Paper",
        "",
        "Dokumen ini berisi aset yang siap ditempel ke paper atau slide.",
        "",
        "## Figure Captions",
        "",
        "**Figure 1. Weak-label benchmark across model families and training epochs.**",
        "The grouped bar chart compares Macro F1 on the weak-label test set for four model families evaluated at 3, 5, and 8 epochs.",
        "",
        "**Figure 2. Gold-subset benchmark across model families and training epochs.**",
        "The grouped bar chart compares Macro F1 on the manually annotated gold subset for the same four model families and three epoch settings.",
        "",
        "## Ready-to-use Takeaway",
        "",
        f"- Weak-label winner: `{weak_winner['Family']}` epoch `{int(weak_winner['Epoch'])}` with Macro F1 `{weak_winner['Weak F1 Macro']:.4f}`.",
        f"- Gold-subset winner: `{gold_winner['Family']}` epoch `{int(gold_winner['Epoch'])}` with Macro F1 `{gold_winner['Gold F1 Macro']:.4f}`.",
        "- Main message: the model that performs best on weak labels is not the same as the model that performs best on the human gold subset.",
        "",
        "## Table",
        "",
        *table_lines,
        "",
        "## Files",
        "",
        "- `model_comparison_weak.png`",
        "- `model_comparison_gold.png`",
        "- `model_comparison_table.csv`",
    ]
    (OUT_DIR / "MODEL_COMPARISON_FOR_PAPER.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = load_combined_frame()
    plot_grouped_bar(
        combined,
        metric="weak_f1_macro",
        title="Weak-label Macro F1 by Model Family and Epoch",
        out_path=OUT_DIR / "model_comparison_weak.png",
    )
    plot_grouped_bar(
        combined,
        metric="gold_f1_macro",
        title="Gold-subset Macro F1 by Model Family and Epoch",
        out_path=OUT_DIR / "model_comparison_gold.png",
    )
    table_df = save_table(combined)
    save_markdown(table_df)
    print(f"Saved paper assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
