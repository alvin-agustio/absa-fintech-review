import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RESOURCES = ROOT / "data" / "resources"


def resolve_reviews_clean_v1() -> Path:
    primary = DATA_PROCESSED / "reviews_clean.csv"
    if primary.exists():
        return primary

    candidates = sorted((DATA_PROCESSED / "archive").rglob("reviews_clean.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError("reviews_clean.csv not found in processed/ or processed/archive/")


def load_whitelist(path: Path) -> set[str]:
    terms = set()
    if not path.exists():
        return terms
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().lower()
        if not line or line.startswith("#"):
            continue
        terms.add(line)
    return terms


def main() -> None:
    lexicon_path = DATA_RESOURCES / "colloquial_lexicon_v2.csv"
    whitelist_path = DATA_RESOURCES / "normalization_whitelist_v2.txt"

    out_candidates = DATA_RESOURCES / "normalization_missing_candidates_v2.csv"
    out_report = DATA_RESOURCES / "normalization_audit_v2.json"

    reviews_path = resolve_reviews_clean_v1()
    reviews = pd.read_csv(reviews_path, usecols=["review_text"])
    lexicon = pd.read_csv(lexicon_path)
    whitelist = load_whitelist(whitelist_path)

    source_set = set(lexicon["source"].astype(str).str.lower())
    target_set = set(lexicon["target"].astype(str).str.lower())
    known_set = source_set | target_set | whitelist

    # Frequent short token scan.
    token_counter = Counter()
    for text in reviews["review_text"].astype(str):
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if len(tok) <= 4:
                token_counter[tok] += 1

    # Common Indonesian words that are not slang candidates.
    stop_like = {
        "yang", "dan", "di", "ke", "dari", "untuk", "ini", "itu", "ada", "saya", "aku", "kami",
        "kita", "mau", "bisa", "akan", "lagi", "baru", "hari", "kali", "dulu", "juga", "sama",
        "dengan", "tidak", "sudah", "belum", "karena", "bagaimana", "kalau", "orang", "sangat",
        "baik", "aman", "data", "akun", "uang", "buat", "coba", "awal", "masa", "atau", "apa",
        "buka", "naik", "lama", "pas", "kok", "lah", "kan", "pun", "ya", "ok", "best", "the",
        "min", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    }

    rows = []
    for tok, freq in token_counter.most_common():
        if freq < 200:
            break
        if tok in known_set or tok in stop_like:
            continue
        if tok.isdigit():
            continue
        rows.append({"token": tok, "freq": int(freq)})

    candidates_df = pd.DataFrame(rows)
    candidates_df.to_csv(out_candidates, index=False, encoding="utf-8")

    # Coverage for existing lexicon source tokens.
    source_hit_rows = []
    for source in sorted(source_set):
        source_hit_rows.append({"source": source, "freq_in_corpus": int(token_counter.get(source, 0))})
    source_hits_df = pd.DataFrame(source_hit_rows).sort_values("freq_in_corpus", ascending=False)

    # Manually tracked ambiguous mapping risks.
    ambiguous_sources = {"dr", "sm", "aja"}
    ambiguous_present = sorted(list(source_set & ambiguous_sources))

    report = {
        "reviews_source": str(reviews_path),
        "n_reviews": int(len(reviews)),
        "lexicon_size": int(len(source_set)),
        "whitelist_size": int(len(whitelist)),
        "n_missing_candidates_freq_ge_200": int(len(candidates_df)),
        "top_20_missing_candidates": candidates_df.head(20).to_dict("records"),
        "top_20_lexicon_sources_by_frequency": source_hits_df.head(20).to_dict("records"),
        "ambiguous_sources_present": ambiguous_present,
        "outputs": {
            "missing_candidates_csv": str(out_candidates),
            "audit_report_json": str(out_report),
        },
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()