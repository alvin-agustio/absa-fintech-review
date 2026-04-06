from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "dataset_absa_v2.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "audits" / "issue_taxonomy"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.aspect_taxonomy import (  # noqa: E402
    ASPECT_DISPLAY_NAMES,
    ASPECT_ORDER,
    GENERAL_ISSUE_LABEL,
    assign_issue_label,
    aspect_presence_hits,
    issue_keywords,
    normalize_text,
)


VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}
STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "ini",
    "itu",
    "untuk",
    "dengan",
    "karena",
    "pada",
    "nya",
    "saya",
    "aku",
    "kami",
    "kita",
    "anda",
    "kalian",
    "the",
    "a",
    "is",
    "are",
    "app",
    "aplikasi",
    "kredivo",
    "akulaku",
    "aja",
    "banget",
    "sangat",
    "lebih",
    "jadi",
    "udah",
    "ga",
    "gak",
    "tidak",
    "nggak",
    "yg",
    "tp",
    "kalo",
    "kalau",
    "atau",
    "dalam",
    "juga",
    "sih",
    "dong",
    "lah",
    "nih",
    "kok",
    "karna",
    "sudah",
    "belum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit issue taxonomy coverage and overlap for the dashboard.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT), help="Path to dataset_absa_v2.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write audit outputs")
    parser.add_argument("--top-n", type=int, default=10, help="How many examples/phrases to keep per aspect")
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {"review_id", "app_name", "review_date", "review_text", "risk_sentiment", "trust_sentiment", "service_sentiment"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return frame


def _clean_sentiment(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().title()
    if text in VALID_SENTIMENTS:
        return text
    return None


def build_long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.copy()
    base["review_id"] = base["review_id"].astype(str)
    base["app_name"] = base["app_name"].astype(str)
    base["review_text"] = base["review_text"].fillna("").astype(str)
    base["review_date"] = pd.to_datetime(base["review_date"], errors="coerce")

    rows: list[dict[str, object]] = []
    for aspect in ASPECT_ORDER:
        sentiment_col = f"{aspect}_sentiment"
        subset = base[["review_id", "app_name", "review_date", "review_text", sentiment_col]].copy()
        subset["pred_label"] = subset[sentiment_col].map(_clean_sentiment)
        subset = subset[subset["pred_label"].notna()].copy()
        subset["aspect"] = aspect
        subset["review_text_clean"] = subset["review_text"].astype(str)
        rows.extend(subset.drop(columns=[sentiment_col]).to_dict("records"))

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return pd.DataFrame(columns=["review_id", "app_name", "review_date", "review_text", "pred_label", "aspect", "review_text_clean"])
    long_df["aspect"] = pd.Categorical(long_df["aspect"], categories=ASPECT_ORDER, ordered=True)
    return long_df.sort_values(["aspect", "review_id"]).reset_index(drop=True)


def extract_salient_phrases(text_series: pd.Series, top_n: int = 3) -> list[str]:
    phrases: list[str] = []
    for text in text_series.fillna("").astype(str):
        tokens = [
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", normalize_text(text))
            if token not in STOPWORDS
        ]
        if len(tokens) < 2:
            continue
        phrases.extend([" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)])
    if not phrases:
        return []
    counts = Counter(phrases)
    return [phrase for phrase, _ in counts.most_common(top_n)]


def shorten_text(text: object, max_chars: int = 160) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def label_issue_series(texts: pd.Series, aspect: str) -> pd.DataFrame:
    labels: list[dict[str, object]] = []
    for text in texts.fillna("").astype(str):
        label, hits = assign_issue_label(text, aspect)
        labels.append(
            {
                "issue": label,
                "matched_keywords": hits,
                "presence_hits": int(aspect_presence_hits(text, aspect)),
            }
        )
    return pd.DataFrame(labels)


def audit_aspect(frame: pd.DataFrame, aspect: str, top_n: int) -> tuple[dict[str, object], pd.DataFrame]:
    aspect_df = frame[(frame["aspect"] == aspect) & (frame["pred_label"] == "Negative")].copy()
    if aspect_df.empty:
        summary = {
            "aspect": aspect,
            "aspect_label": ASPECT_DISPLAY_NAMES[aspect],
            "negative_rows": 0,
            "specific_rows": 0,
            "generic_rows": 0,
            "specific_coverage_pct": 0.0,
            "top_issue": GENERAL_ISSUE_LABEL,
            "top_issue_count": 0,
            "top_issue_share_pct": 0.0,
            "aspect_presence_hit_rate_pct": 0.0,
            "generic_top_phrases": [],
        }
        return summary, pd.DataFrame()

    issue_meta = label_issue_series(aspect_df["review_text_clean"], aspect)
    aspect_df = aspect_df.reset_index(drop=True).join(issue_meta)

    issue_counts = aspect_df["issue"].value_counts()
    specific = aspect_df[aspect_df["issue"] != GENERAL_ISSUE_LABEL].copy()
    generic = aspect_df[aspect_df["issue"] == GENERAL_ISSUE_LABEL].copy()

    top_specific_issue = GENERAL_ISSUE_LABEL
    top_specific_count = 0
    if not specific.empty:
        top_specific_issue = str(specific["issue"].value_counts().idxmax())
        top_specific_count = int(specific["issue"].value_counts().iloc[0])

    top_issue = str(issue_counts.idxmax())
    top_issue_count = int(issue_counts.iloc[0])
    total = int(len(aspect_df))

    generic_top_phrases = extract_salient_phrases(generic["review_text_clean"], top_n=top_n)

    generic_examples = generic.copy()
    generic_examples["review_text_len"] = generic_examples["review_text_clean"].str.len()
    generic_examples["presence_hits"] = pd.to_numeric(generic_examples["presence_hits"], errors="coerce").fillna(0).astype(int)
    generic_examples = generic_examples.sort_values(
        ["presence_hits", "review_text_len", "review_id"],
        ascending=[False, False, True],
    )

    example_rows = []
    for _, row in generic_examples.head(top_n).iterrows():
        example_rows.append(
            {
                "aspect": aspect,
                "aspect_label": ASPECT_DISPLAY_NAMES[aspect],
                "review_id": row["review_id"],
                "app_name": row["app_name"],
                "issue": row["issue"],
                "presence_hits": int(row["presence_hits"]),
                "matched_keywords": ", ".join(row["matched_keywords"]) if row["matched_keywords"] else "-",
                "snippet": shorten_text(row["review_text_clean"], max_chars=180),
            }
        )

    summary = {
        "aspect": aspect,
        "aspect_label": ASPECT_DISPLAY_NAMES[aspect],
        "negative_rows": total,
        "specific_rows": int(len(specific)),
        "generic_rows": int(len(generic)),
        "specific_coverage_pct": round((len(specific) / total * 100.0), 1) if total else 0.0,
        "top_issue": top_issue,
        "top_issue_count": top_issue_count,
        "top_issue_share_pct": round(top_issue_count / total * 100.0, 1) if total else 0.0,
        "top_specific_issue": top_specific_issue,
        "top_specific_issue_count": top_specific_count,
        "aspect_presence_hit_rate_pct": round(
            float((aspect_df["presence_hits"] > 0).mean()) * 100.0, 1
        ) if total else 0.0,
        "generic_top_phrases": generic_top_phrases,
    }
    return summary, pd.DataFrame(example_rows)


def build_overlap_table() -> pd.DataFrame:
    kw_map: dict[str, set[str]] = defaultdict(set)
    for aspect in ASPECT_ORDER:
        for keyword in issue_keywords(aspect):
            kw_map[normalize_text(keyword)].add(aspect)

    rows: list[dict[str, object]] = []
    for keyword, aspects in kw_map.items():
        if len(aspects) <= 1:
            continue
        rows.append(
            {
                "keyword": keyword,
                "aspect_count": len(aspects),
                "aspects": ", ".join(sorted(aspects)),
            }
        )
    return pd.DataFrame(rows).sort_values(["aspect_count", "keyword"], ascending=[False, True]).reset_index(drop=True)


def build_pair_overlap_table() -> pd.DataFrame:
    aspect_keywords = {aspect: {normalize_text(k) for k in issue_keywords(aspect)} for aspect in ASPECT_ORDER}
    rows: list[dict[str, object]] = []
    for left, right in combinations(ASPECT_ORDER, 2):
        shared = sorted(aspect_keywords[left] & aspect_keywords[right])
        rows.append(
            {
                "aspect_left": ASPECT_DISPLAY_NAMES[left],
                "aspect_right": ASPECT_DISPLAY_NAMES[right],
                "shared_count": len(shared),
                "shared_keywords": ", ".join(shared[:30]),
            }
        )
    return pd.DataFrame(rows)


def build_recommendations(summary_rows: list[dict[str, object]], overlap_df: pd.DataFrame) -> list[str]:
    recs: list[str] = []
    max_generic = max((row["generic_rows"] for row in summary_rows), default=0)
    if max_generic > 0:
        recs.append("Audit bucket umum paling banyak di aspek dengan generic_rows tertinggi, lalu pecah frasa yang paling sering muncul.")

    if any(row["specific_coverage_pct"] < 60 for row in summary_rows):
        recs.append("Tambah keyword/frasa untuk issue yang masih sering jatuh ke Belum cukup spesifik.")

    if not overlap_df.empty and len(overlap_df) > 0:
        top_overlap = overlap_df.iloc[0]
        recs.append(
            f"Kurangi overlap keyword bersama, terutama keyword yang muncul di {top_overlap['aspects']}."
        )

    recs.append("Pertahankan corpus besar sebagai bahan audit taxonomy, lalu refine issue rules sebelum menambah corpus baru.")
    return recs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_frame(input_path)
    long_df = build_long_frame(raw)
    if long_df.empty:
        raise RuntimeError("No valid long-form ABSA rows found in the input corpus.")

    summary_rows: list[dict[str, object]] = []
    generic_examples: list[pd.DataFrame] = []

    for aspect in ASPECT_ORDER:
        summary, examples_df = audit_aspect(long_df, aspect, args.top_n)
        summary_rows.append(summary)
        if not examples_df.empty:
            generic_examples.append(examples_df)

    summary_df = pd.DataFrame(summary_rows)
    if generic_examples:
        examples_df = pd.concat(generic_examples, ignore_index=True)
    else:
        examples_df = pd.DataFrame(
            columns=[
                "aspect",
                "aspect_label",
                "review_id",
                "app_name",
                "issue",
                "presence_hits",
                "matched_keywords",
                "snippet",
            ]
        )
    overlap_df = build_overlap_table()
    pair_overlap_df = build_pair_overlap_table()
    recommendations = build_recommendations(summary_rows, overlap_df)

    report = {
        "input_csv": str(input_path),
        "rows_loaded": int(len(raw)),
        "rows_long": int(len(long_df)),
        "generic_examples_rows": int(len(examples_df)),
        "summary": summary_rows,
        "overlap": overlap_df.head(50).to_dict("records"),
        "pair_overlap": pair_overlap_df.to_dict("records"),
        "recommendations": recommendations,
    }

    summary_path = output_dir / "issue_taxonomy_summary.csv"
    generic_path = output_dir / "issue_taxonomy_generic_candidates.csv"
    overlap_path = output_dir / "issue_taxonomy_overlap.csv"
    pair_overlap_path = output_dir / "issue_taxonomy_pair_overlap.csv"
    report_path = output_dir / "issue_taxonomy_report.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    examples_df.to_csv(generic_path, index=False, encoding="utf-8")
    overlap_df.to_csv(overlap_path, index=False, encoding="utf-8")
    pair_overlap_df.to_csv(pair_overlap_path, index=False, encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Loaded corpus: {input_path}")
    print(f"[OK] Long rows: {len(long_df):,}")
    print(f"[OK] Wrote: {summary_path}")
    print(f"[OK] Wrote: {generic_path}")
    print(f"[OK] Wrote: {overlap_path}")
    print(f"[OK] Wrote: {pair_overlap_path}")
    print(f"[OK] Wrote: {report_path}")
    print()
    print("Per-aspect audit:")
    for row in summary_rows:
        print(
            f"- {row['aspect_label']}: specific coverage {row['specific_coverage_pct']:.1f}% | "
            f"top issue {row['top_issue']} | generic rows {row['generic_rows']}"
        )
    print()
    print("Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")


if __name__ == "__main__":
    main()
