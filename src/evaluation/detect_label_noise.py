import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from config import DATA_PROCESSED


def main():
    parser = argparse.ArgumentParser(
        description="Detect candidate noisy labels from MC Dropout outputs"
    )
    parser.add_argument("--input_csv", default=str(DATA_PROCESSED / "uncertainty" / "mc_predictions.csv"))
    parser.add_argument("--output_dir", default=str(DATA_PROCESSED / "noise"))
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    threshold = df[args.uncertainty_col].quantile(args.high_uncertainty_quantile)

    df["is_high_uncertainty"] = df[args.uncertainty_col] >= threshold
    df["is_mismatch"] = df["pred_label"] != df["weak_label"]
    df["is_noisy_candidate"] = df["is_high_uncertainty"] & df["is_mismatch"]

    clean_df = df[~df["is_noisy_candidate"]].copy()
    noisy_df = df[df["is_noisy_candidate"]].copy()

    df.to_csv(output_dir / "mc_with_noise_flags.csv", index=False)
    clean_df.to_csv(output_dir / "clean_data.csv", index=False)
    noisy_df.to_csv(output_dir / "noisy_data.csv", index=False)

    summary = {
        "n_total": int(len(df)),
        "n_noisy_candidates": int(len(noisy_df)),
        "n_clean": int(len(clean_df)),
        "noise_ratio": float(len(noisy_df) / len(df)) if len(df) else 0.0,
        "uncertainty_col": args.uncertainty_col,
        "uncertainty_threshold": float(threshold),
        "high_uncertainty_quantile": float(args.high_uncertainty_quantile),
    }

    with open(output_dir / "noise_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[NOISE] Detection selesai.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
