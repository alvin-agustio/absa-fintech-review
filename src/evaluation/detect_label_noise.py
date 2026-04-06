import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from config import DATA_PROCESSED


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_model_family(df: pd.DataFrame, explicit_family: str | None) -> str:
    if explicit_family:
        return explicit_family.strip().lower().replace(" ", "_")
    if "model_family" in df.columns and df["model_family"].notna().any():
        return str(df["model_family"].dropna().iloc[0]).strip().lower().replace(" ", "_")
    return "baseline"


def infer_run_name(input_csv: Path, df: pd.DataFrame, explicit_run_name: str | None) -> str:
    if explicit_run_name:
        return explicit_run_name.strip().replace(" ", "_")
    if "run_name" in df.columns and df["run_name"].notna().any():
        return str(df["run_name"].dropna().iloc[0]).strip().replace(" ", "_")
    return input_csv.parent.name if input_csv.parent != input_csv.parent.parent else input_csv.stem


def resolve_input_csv(input_csv: str | None, input_dir: str | None) -> Path:
    if input_csv:
        return Path(input_csv)
    if input_dir:
        return Path(input_dir) / "mc_predictions.csv"
    return DATA_PROCESSED / "uncertainty" / "mc_predictions.csv"


def resolve_output_dir(*, output_dir: str | None, model_family: str, run_name: str) -> Path:
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = DATA_PROCESSED / "noise"
    return base_dir / model_family / run_name


def compute_thresholds(
    df: pd.DataFrame,
    uncertainty_col: str,
    quantile: float,
    threshold_scope: str,
) -> tuple[pd.Series, dict[str, float]]:
    thresholds: dict[str, float] = {}
    if threshold_scope == "global":
        threshold = float(df[uncertainty_col].quantile(quantile))
        thresholds["global"] = threshold
        return pd.Series(threshold, index=df.index, dtype=float), thresholds

    threshold_series = pd.Series(index=df.index, dtype=float)
    for aspect, part in df.groupby("aspect"):
        threshold = float(part[uncertainty_col].quantile(quantile))
        thresholds[str(aspect)] = threshold
        threshold_series.loc[part.index] = threshold
    return threshold_series, thresholds


def summarize_noise(df: pd.DataFrame, noisy_col: str) -> dict:
    out = {
        "n_rows": int(len(df)),
        "n_reviews": int(df["review_id"].nunique()) if "review_id" in df.columns else None,
        "n_noisy_candidates": int(df[noisy_col].sum()),
        "n_clean": int((~df[noisy_col]).sum()),
        "noise_ratio": float(df[noisy_col].mean()) if len(df) else 0.0,
    }
    if "aspect" in df.columns:
        out["by_aspect"] = {}
        for aspect, part in df.groupby("aspect"):
            out["by_aspect"][str(aspect)] = {
                "n_rows": int(len(part)),
                "n_noisy_candidates": int(part[noisy_col].sum()),
                "noise_ratio": float(part[noisy_col].mean()) if len(part) else 0.0,
            }
    return out


def main():
    parser = argparse.ArgumentParser(description="Detect candidate noisy labels from MC Dropout outputs")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_family", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--uncertainty_col",
        default="uncertainty_entropy",
        choices=["uncertainty_entropy", "uncertainty_variance"],
    )
    parser.add_argument(
        "--high_uncertainty_quantile",
        type=float,
        default=0.8,
        help="Quantile threshold untuk high-uncertainty (contoh 0.8 = top 20%)",
    )
    parser.add_argument(
        "--threshold_scope",
        default="global",
        choices=["global", "per_aspect"],
        help="Gunakan threshold uncertainty global atau per aspek.",
    )
    args = parser.parse_args()

    input_csv_path = resolve_input_csv(args.input_csv, args.input_dir).expanduser().resolve()
    df = pd.read_csv(input_csv_path)

    required_cols = {"review_id", "aspect", "weak_label", "pred_label", args.uncertainty_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"MC prediction file missing required columns: {sorted(missing)}")

    df = df.copy()
    df = df[df[args.uncertainty_col].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid rows found after dropping NaN in {args.uncertainty_col}.")

    model_family = infer_model_family(df, args.model_family)
    run_name = infer_run_name(input_csv_path, df, args.run_name)
    output_dir = resolve_output_dir(output_dir=args.output_dir, model_family=model_family, run_name=run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_series, thresholds = compute_thresholds(
        df=df,
        uncertainty_col=args.uncertainty_col,
        quantile=args.high_uncertainty_quantile,
        threshold_scope=args.threshold_scope,
    )

    df["uncertainty_threshold"] = threshold_series
    df["is_high_uncertainty"] = df[args.uncertainty_col] >= df["uncertainty_threshold"]
    df["is_mismatch"] = df["pred_label"] != df["weak_label"]
    df["is_noisy_candidate"] = df["is_high_uncertainty"] & df["is_mismatch"]

    clean_df = df[~df["is_noisy_candidate"]].copy()
    noisy_df = df[df["is_noisy_candidate"]].copy()

    df.to_csv(output_dir / "mc_with_noise_flags.csv", index=False)
    clean_df.to_csv(output_dir / "clean_data.csv", index=False)
    noisy_df.to_csv(output_dir / "noisy_data.csv", index=False)

    summary = {
        "generated_at_utc": iso_utc_now(),
        "input_csv": str(input_csv_path),
        "model_family": model_family,
        "run_name": run_name,
        "uncertainty_col": args.uncertainty_col,
        "high_uncertainty_quantile": float(args.high_uncertainty_quantile),
        "threshold_scope": args.threshold_scope,
        "uncertainty_thresholds": thresholds,
        "n_total": int(len(df)),
        "n_noisy_candidates": int(len(noisy_df)),
        "n_clean": int(len(clean_df)),
        "noise_ratio": float(len(noisy_df) / len(df)) if len(df) else 0.0,
        "mismatch_rate": float(df["is_mismatch"].mean()) if len(df) else 0.0,
        "high_uncertainty_rate": float(df["is_high_uncertainty"].mean()) if len(df) else 0.0,
        "full_stats": summarize_noise(df, "is_noisy_candidate"),
        "clean_stats": summarize_noise(clean_df.assign(is_noisy_candidate=False), "is_noisy_candidate"),
        "noisy_stats": summarize_noise(noisy_df.assign(is_noisy_candidate=True), "is_noisy_candidate"),
    }

    with open(output_dir / "noise_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[NOISE] Detection selesai.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
