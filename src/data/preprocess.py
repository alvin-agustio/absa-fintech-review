"""
Preprocessing Pipeline
=======================
Clean raw review text: dedup, remove URLs/emojis, lowercase, normalize,
remove short reviews. Preserves punctuation and slang.

Usage:
    python preprocess.py
"""

import re

import pandas as pd

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from config import DATA_RAW, DATA_PROCESSED


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
    """Remove mathematical/styled unicode characters."""
    return "".join(ch for ch in text if not (0x1D400 <= ord(ch) <= 0x1D7FF))


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for a single text."""
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_unicode_artifacts(text)
    text = remove_newlines(text)
    text = text.lower()
    text = normalize_whitespace(text)
    return text


def main():
    input_path = DATA_RAW / "reviews_raw.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Raw data not found: {input_path}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"[PREP] Loaded {len(df)} raw reviews")

    # Step 1: Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["review_text_raw"], keep="first").copy()
    print(f"[PREP] Removed {before - len(df)} duplicates → {len(df)}")

    # Step 2: Clean text
    df["review_text"] = df["review_text_raw"].astype(str).apply(preprocess_text)

    # Step 3: Remove empty reviews
    before = len(df)
    df = df[df["review_text"].str.len() > 0].copy()
    print(f"[PREP] Removed {before - len(df)} empty reviews → {len(df)}")

    # Step 4: Remove short reviews (< 3 words)
    before = len(df)
    df = df[df["review_text"].str.split().str.len() >= 3].copy()
    print(f"[PREP] Removed {before - len(df)} short reviews → {len(df)}")

    # Reset index and assign review_id
    df = df.reset_index(drop=True)
    df["review_id"] = range(1, len(df) + 1)

    # Select output columns
    df = df[["review_id", "app_name", "rating", "review_date", "review_text"]]

    output_path = DATA_PROCESSED / "reviews_clean.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n[PREP] Saved {len(df)} clean reviews to {output_path}")
    print(f"  Kredivo: {len(df[df['app_name'] == 'Kredivo'])}")
    print(f"  Akulaku: {len(df[df['app_name'] == 'Akulaku'])}")


if __name__ == "__main__":
    main()
