from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA_PROCESSED  # noqa: E402
from src.evaluation.epoch_protocol import (  # noqa: E402
    format_recommendation_report,
    load_epoch_sweep,
    recommend_epochs,
)


def find_default_csv() -> Path:
    candidates = [
        DATA_PROCESSED / "evaluation" / "epoch_comparison_summary.csv",
        ROOT / "droplet" / "skripsi_eval_core" / "data" / "processed" / "evaluation" / "epoch_comparison_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No epoch_comparison_summary.csv found in the active or legacy evaluation directories."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend the best epoch per family from an existing epoch sweep"
    )
    parser.add_argument("--input_csv", default=None, help="Path to epoch_comparison_summary.csv")
    parser.add_argument("--output_json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    csv_path = Path(args.input_csv) if args.input_csv else find_default_csv()
    df = load_epoch_sweep(csv_path)
    summary = recommend_epochs(df)

    print(format_recommendation_report(summary))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Wrote recommendation JSON to {output_path}")


if __name__ == "__main__":
    main()
