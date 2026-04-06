from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "dataset_absa_50k_v2_intersection.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "audits" / "insight_layer"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_issue_taxonomy import build_long_frame, load_frame  # noqa: E402
from src.dashboard.aspect_taxonomy import (  # noqa: E402
    ASPECT_DISPLAY_NAMES,
    ASPECT_ORDER,
    GENERAL_ISSUE_LABEL,
    assign_issue_label,
)
from src.dashboard.summary_rules import build_summary_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dashboard insight layer without using an LLM.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT), help="Path to the ABSA review-level CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for audit artifacts.")
    parser.add_argument("--top-examples", type=int, default=5, help="Examples per aspect/app for manual review.")
    return parser.parse_args()


def attach_issue_labels(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df.copy()

    frame = long_df.copy()
    labels: list[str] = []
    hits_payload: list[str] = []
    for row in frame.itertuples(index=False):
        label, hits = assign_issue_label(getattr(row, "review_text_clean", ""), getattr(row, "aspect", ""))
        labels.append(label)
        hits_payload.append(", ".join(hits) if hits else "-")
    frame["issue"] = labels
    frame["matched_keywords"] = hits_payload
    frame["review_id_ext"] = frame["review_id"].astype(str)
    return frame


def build_score_table(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "aspect",
                "positive_share",
                "neutral_share",
                "negative_share",
                "dominant_sentiment",
                "dominant_share",
                "rows",
            ]
        )

    pivot = (
        long_df.groupby(["aspect", "pred_label"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "count"})
        .pivot(index="aspect", columns="pred_label", values="count")
        .reindex(index=ASPECT_ORDER, columns=["Positive", "Neutral", "Negative"], fill_value=0)
        .fillna(0)
    )
    totals = pivot.sum(axis=1).replace(0, pd.NA)
    shares = pivot.div(totals, axis=0).fillna(0.0)
    dominant_sentiment = shares.idxmax(axis=1)
    dominant_share = shares.max(axis=1)
    out = pd.DataFrame(
        {
            "aspect": pivot.index,
            "positive_share": (shares["Positive"] * 100).round(1),
            "neutral_share": (shares["Neutral"] * 100).round(1),
            "negative_share": (shares["Negative"] * 100).round(1),
            "dominant_sentiment": dominant_sentiment,
            "dominant_share": (dominant_share * 100).round(1),
            "rows": pivot.sum(axis=1).astype(int),
        }
    ).reset_index(drop=True)
    out["aspect"] = pd.Categorical(out["aspect"], categories=ASPECT_ORDER, ordered=True)
    return out.sort_values("aspect").reset_index(drop=True)


def expected_signal_aspect(score_df: pd.DataFrame) -> str | None:
    if score_df.empty:
        return None
    row = score_df.sort_values(["negative_share", "positive_share"], ascending=[False, False]).iloc[0]
    return str(row["aspect"])


def expected_best_aspect(score_df: pd.DataFrame) -> str | None:
    if score_df.empty:
        return None
    row = score_df.sort_values(["positive_share", "neutral_share"], ascending=[False, False]).iloc[0]
    return str(row["aspect"])


def extract_summary_audit(payload: dict[str, Any], score_df: pd.DataFrame) -> dict[str, Any]:
    overall = payload.get("overall", {}) if isinstance(payload, dict) else {}
    signal = payload.get("signal", payload.get("focus", {})) if isinstance(payload, dict) else {}
    meaning = payload.get("meaning", {}) if isinstance(payload, dict) else {}
    apps = payload.get("apps", []) if isinstance(payload, dict) else []

    overall_metrics = overall.get("metrics", {}) if isinstance(overall, dict) else {}
    signal_metrics = signal.get("metrics", {}) if isinstance(signal, dict) else {}
    meaning_metrics = meaning.get("metrics", {}) if isinstance(meaning, dict) else {}

    expected_focus = expected_signal_aspect(score_df)
    expected_best = expected_best_aspect(score_df)
    reported_focus = str(signal_metrics.get("aspect")) if signal_metrics.get("aspect") is not None else None
    reported_best = str(overall_metrics.get("best_aspect")) if overall_metrics.get("best_aspect") is not None else None
    reported_worst = str(overall_metrics.get("worst_aspect")) if overall_metrics.get("worst_aspect") is not None else None

    return {
        "status": payload.get("status"),
        "quality": payload.get("coverage", {}).get("quality"),
        "warnings": payload.get("warnings", []),
        "expected_focus_aspect": expected_focus,
        "reported_focus_aspect": reported_focus,
        "focus_matches_negative_leader": reported_focus == expected_focus,
        "expected_best_aspect": expected_best,
        "reported_best_aspect": reported_best,
        "best_matches_positive_leader": reported_best == expected_best,
        "reported_worst_aspect": reported_worst,
        "meaning_worst_aspect": meaning_metrics.get("worst_aspect"),
        "app_cards": len(apps) if isinstance(apps, list) else 0,
        "overall_text": overall.get("body", overall.get("text", "")) if isinstance(overall, dict) else "",
        "signal_text": signal.get("body", signal.get("text", "")) if isinstance(signal, dict) else "",
        "meaning_text": meaning.get("body", meaning.get("text", "")) if isinstance(meaning, dict) else "",
    }


def issue_snapshot(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["aspect", "top_issue", "issue_share_pct", "generic_share_pct", "negative_rows"])

    rows: list[dict[str, Any]] = []
    for aspect in ASPECT_ORDER:
        subset = long_df[(long_df["aspect"] == aspect) & (long_df["pred_label"] == "Negative")].copy()
        if subset.empty:
            rows.append(
                {
                    "aspect": aspect,
                    "top_issue": GENERAL_ISSUE_LABEL,
                    "issue_share_pct": 0.0,
                    "generic_share_pct": 0.0,
                    "negative_rows": 0,
                }
            )
            continue
        counts = subset["issue"].value_counts()
        top_issue = str(counts.index[0])
        top_share = round(float(counts.iloc[0]) / float(counts.sum()) * 100.0, 1)
        generic_share = round(
            float(subset["issue"].eq(GENERAL_ISSUE_LABEL).sum()) / float(len(subset)) * 100.0,
            1,
        )
        rows.append(
            {
                "aspect": aspect,
                "top_issue": top_issue,
                "issue_share_pct": top_share,
                "generic_share_pct": generic_share,
                "negative_rows": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def manual_review_examples(long_df: pd.DataFrame, top_examples: int) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            columns=["scope", "app_name", "aspect", "pred_label", "issue", "review_date", "matched_keywords", "review_text"]
        )

    rows: list[pd.DataFrame] = []
    for aspect in ASPECT_ORDER:
        subset = long_df[long_df["aspect"] == aspect].copy()
        for sentiment in ["Positive", "Neutral", "Negative"]:
            sent_df = subset[subset["pred_label"] == sentiment].copy()
            if sent_df.empty:
                continue
            picked = sent_df.sort_values(["review_date", "review_id"], ascending=[False, True]).head(top_examples).copy()
            picked["scope"] = f"all::{aspect}::{sentiment.lower()}"
            rows.append(picked)
        for app_name in sorted(subset["app_name"].dropna().astype(str).unique().tolist()):
            app_neg = subset[(subset["app_name"] == app_name) & (subset["pred_label"] == "Negative")].copy()
            if app_neg.empty:
                continue
            picked = app_neg.sort_values(["review_date", "review_id"], ascending=[False, True]).head(top_examples).copy()
            picked["scope"] = f"{app_name.lower()}::{aspect}::negative"
            rows.append(picked)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["review_text"] = out["review_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return out[
        ["scope", "app_name", "aspect", "pred_label", "issue", "review_date", "matched_keywords", "review_text"]
    ].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_frame(input_path)
    long_df = attach_issue_labels(build_long_frame(raw_df))
    score_df = build_score_table(long_df)
    payload = build_summary_payload(long_df, score_df=score_df)
    summary_audit = extract_summary_audit(payload, score_df)

    score_df.to_csv(output_dir / "summary_score_table.csv", index=False, encoding="utf-8")
    issue_snapshot_df = issue_snapshot(long_df)
    issue_snapshot_df.to_csv(output_dir / "issue_snapshot.csv", index=False, encoding="utf-8")

    app_rows: list[dict[str, Any]] = []
    for app_name in sorted(long_df["app_name"].dropna().astype(str).unique().tolist()):
        app_frame = long_df[long_df["app_name"] == app_name].copy()
        app_score_df = build_score_table(app_frame)
        app_payload = build_summary_payload(app_frame, score_df=app_score_df)
        app_audit = extract_summary_audit(app_payload, app_score_df)
        app_rows.append({"app_name": app_name, **app_audit})
    app_audit_df = pd.DataFrame(app_rows)
    app_audit_df.to_csv(output_dir / "app_summary_audit.csv", index=False, encoding="utf-8")

    examples_df = manual_review_examples(long_df, top_examples=args.top_examples)
    examples_df.to_csv(output_dir / "manual_review_examples.csv", index=False, encoding="utf-8")

    report = {
        "input_csv": str(input_path),
        "rows_long": int(len(long_df)),
        "unique_reviews": int(long_df["review_id"].nunique()) if not long_df.empty else 0,
        "apps": sorted(long_df["app_name"].dropna().astype(str).unique().tolist()) if not long_df.empty else [],
        "summary_audit": summary_audit,
        "issue_snapshot": issue_snapshot_df.to_dict(orient="records"),
        "app_summary_audit": app_rows,
        "manual_review_examples_rows": int(len(examples_df)),
    }
    (output_dir / "insight_layer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Insight audit written to: {output_dir}")
    print(
        "[SUMMARY] "
        f"status={summary_audit['status']} "
        f"quality={summary_audit['quality']} "
        f"focus_match={summary_audit['focus_matches_negative_leader']} "
        f"best_match={summary_audit['best_matches_positive_leader']}"
    )
    for row in issue_snapshot_df.itertuples(index=False):
        print(
            f"[ISSUE] {ASPECT_DISPLAY_NAMES[str(row.aspect)]}: "
            f"top={row.top_issue} share={row.issue_share_pct:.1f}% generic={row.generic_share_pct:.1f}%"
        )


if __name__ == "__main__":
    main()
