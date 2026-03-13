import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RESOURCES = ROOT / "data" / "resources"


def resolve_input_path(filename: str) -> Path:
    primary = DATA_PROCESSED / filename
    if primary.exists():
        return primary

    archive_root = DATA_PROCESSED / "archive"
    if archive_root.exists():
        candidates = sorted(archive_root.rglob(filename), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"Could not locate input file '{filename}' in processed/ or processed/archive/")


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2b55"
        "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_newlines(text: str) -> str:
    return re.sub(r"[\r\n]+", " ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


def remove_unicode_artifacts(text: str) -> str:
    return "".join(ch for ch in text if not (0x1D400 <= ord(ch) <= 0x1D7FF))


def load_lexicon(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"source", "target"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Lexicon file must contain columns {sorted(required)}: {path}")

    lexicon = {}
    for _, row in df.iterrows():
        source = str(row["source"]).strip().lower()
        target = str(row["target"]).strip().lower()
        if source and target:
            lexicon[source] = target
    return lexicon


def load_whitelist(path: Path) -> set[str]:
    if not path.exists():
        return set()

    terms = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().lower()
        if not line or line.startswith("#"):
            continue
        terms.add(line)
    return terms


def normalize_slang(text: str, slang_map: dict[str, str], whitelist: set[str]) -> tuple[str, int]:
    tokens = text.split()
    replacements = 0
    normalized_tokens = []
    for tok in tokens:
        if tok in whitelist:
            normalized_tokens.append(tok)
            continue
        repl = slang_map.get(tok)
        if repl is not None and repl != tok:
            normalized_tokens.append(repl)
            replacements += 1
        else:
            normalized_tokens.append(tok)
    return " ".join(normalized_tokens), replacements


def preprocess_v2(text: str, slang_map: dict[str, str], whitelist: set[str]) -> tuple[str, int]:
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_unicode_artifacts(text)
    text = remove_newlines(text)
    text = text.lower()
    text = normalize_whitespace(text)
    text, replacements = normalize_slang(text, slang_map, whitelist)
    text = normalize_whitespace(text)
    return text, replacements


def is_junk_text(text: str) -> bool:
    compact = text.replace(" ", "")
    if len(compact) <= 3:
        return True
    if re.fullmatch(r"[\W_]+", compact):
        return True
    return False


def main() -> None:
    lexicon_path = DATA_RESOURCES / "colloquial_lexicon_v2.csv"
    whitelist_path = DATA_RESOURCES / "normalization_whitelist_v2.txt"

    reviews_clean_path = resolve_input_path("reviews_clean.csv")
    dataset_absa_path = resolve_input_path("dataset_absa.csv")

    reviews_v2_path = DATA_PROCESSED / "reviews_clean_v2.csv"
    dataset_absa_v2_path = DATA_PROCESSED / "dataset_absa_v2.csv"
    report_path = DATA_PROCESSED / "dataset_absa_v2_report.json"

    reviews = pd.read_csv(reviews_clean_path)
    absa_v1 = pd.read_csv(dataset_absa_path)

    slang_map = load_lexicon(lexicon_path)
    whitelist = load_whitelist(whitelist_path)

    before_reviews = len(reviews)

    reviews = reviews.copy()
    normalized = reviews["review_text"].astype(str).apply(lambda x: preprocess_v2(x, slang_map, whitelist))
    reviews["review_text"] = normalized.apply(lambda x: x[0])
    reviews["_replacements"] = normalized.apply(lambda x: int(x[1]))
    total_replacements = int(reviews["_replacements"].sum())
    rows_with_replacements = int((reviews["_replacements"] > 0).sum())

    reviews = reviews[reviews["review_text"].str.split().str.len() >= 3].copy()
    reviews = reviews[~reviews["review_text"].apply(is_junk_text)].copy()

    # Important: dedup after normalization to remove post-clean duplicates.
    reviews = reviews.drop_duplicates(subset=["app_name", "review_text"], keep="first").copy()
    reviews = reviews.reset_index(drop=True)
    reviews = reviews.drop(columns=["_replacements"], errors="ignore")

    reviews.to_csv(reviews_v2_path, index=False, encoding="utf-8")

    absa_v1 = absa_v1.copy()
    absa_v1["review_id"] = absa_v1["review_id"].astype(str)
    reviews["review_id"] = reviews["review_id"].astype(str)

    merge_cols = [
        "review_id",
        "app_name",
        "rating",
        "review_date",
        "aspect_type",
        "risk_sentiment",
        "trust_sentiment",
        "service_sentiment",
        "reasoning",
    ]

    merged = reviews[["review_id", "review_text"]].merge(
        absa_v1[merge_cols],
        on="review_id",
        how="inner",
    )

    merged = merged[
        [
            "review_id",
            "app_name",
            "rating",
            "review_date",
            "review_text",
            "aspect_type",
            "risk_sentiment",
            "trust_sentiment",
            "service_sentiment",
            "reasoning",
        ]
    ]

    merged.to_csv(dataset_absa_v2_path, index=False, encoding="utf-8")

    report = {
        "normalization_lexicon_path": str(lexicon_path),
        "normalization_whitelist_path": str(whitelist_path),
        "normalization_lexicon_size": int(len(slang_map)),
        "normalization_whitelist_size": int(len(whitelist)),
        "normalization_total_replacements": total_replacements,
        "normalization_rows_with_replacements": rows_with_replacements,
        "reviews_clean_v1_rows": int(before_reviews),
        "reviews_clean_v2_rows": int(len(reviews)),
        "reviews_dropped_in_v2": int(before_reviews - len(reviews)),
        "dataset_absa_v1_rows": int(len(absa_v1)),
        "dataset_absa_v2_rows": int(len(merged)),
        "dataset_absa_rows_dropped_by_intersection": int(len(absa_v1) - len(merged)),
        "shared_review_ids": int(merged["review_id"].nunique()),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("[OK] Wrote:", reviews_v2_path)
    print("[OK] Wrote:", dataset_absa_v2_path)
    print("[OK] Wrote:", report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()