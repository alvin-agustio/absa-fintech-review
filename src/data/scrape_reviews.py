"""
Google Play Store Review Scraper & Preprocessor
for Aspect-Based Sentiment Analysis (ABSA)
=============================================
Apps: Kredivo & Akulaku
Aspects: Risk (risiko) & Trust (kepercayaan)
Sentiment: Positive / Negative / Neutral / Unlabeled

Output: dataset_kredivo_akulaku_absa.csv + pipeline_report.md
"""

import re
import pandas as pd
from datetime import datetime
from google_play_scraper import Sort, reviews

# ── CONFIG ────────────────────────────────────────────────────────────
APPS = {
    "Kredivo": "com.finaccel.android",
    "Akulaku": "io.silvrr.installment",
}
LANG = "id"
COUNTRY = "id"
REVIEWS_PER_APP = 1000  # target per app (akan di-filter duplicate)
OUTPUT_CSV = "dataset_kredivo_akulaku_absa.csv"
OUTPUT_MD = "pipeline_report.md"

# ── KEYWORD LISTS ─────────────────────────────────────────────────────
RISK_KW = [
    "bunga",
    "denda",
    "penagihan",
    "penagih",
    "tagihan",
    "dc ",
    "debt collector",
    "debt",
    "collector",
    "keamanan",
    "data",
    "ancaman",
    "ancam",
    "diancam",
    "mengancam",
    "teror",
    "diteror",
    "spam",
    "telat",
    "terlambat",
    "keterlambatan",
    "telpon",
    "nelpon",
    "ditelpon",
    "telepon",
    "ditelepon",
    "bocor",
    "sebar",
    "disebar",
    "tersebar",
    "limit",
    "skor",
    "score",
    "diblokir",
    "blokir",
    "ditolak",
    "tolak",
    "gagal",
    "tidak bisa",
    "gak bisa",
    "ga bisa",
    "gabisa",
    "gbisa",
    "gk bisa",
    "tdk bisa",
    "cicilan",
    "angsuran",
    "tenor",
    "jatuh tempo",
    "pinjam",
    "pinjaman",
    "pencairan",
    "cair",
    "riba",
    "lintah",
    "mencekik",
    "mahal",
    "admin",
    "potongan",
    "biaya",
]

TRUST_KW = [
    "percaya",
    "terpercaya",
    "kepercayaan",
    "amanah",
    "aman",
    "legal",
    "legalitas",
    "ojk",
    "resmi",
    "transparan",
    "transparansi",
    "jelas",
    "pelayanan",
    "layanan",
    "cs ",
    "customer service",
    "respon",
    "responsif",
    "response",
    "mudah",
    "cepat",
    "proses",
    "prosesnya",
    "membantu",
    "memudahkan",
    "memuaskan",
    "puas",
    "bagus",
    "baik",
    "mantap",
    "keren",
    "top",
    "best",
    "rekomendasi",
    "recommended",
    "rekomended",
    "aplikasi",
    "apk",
    "app",
    "eror",
    "error",
    "bug",
    "lemot",
    "lelet",
    "penipuan",
    "nipu",
    "tipu",
    "scam",
    "hoax",
    "bohong",
]

POSITIVE_KW = [
    "bagus",
    "baik",
    "mantap",
    "keren",
    "top",
    "best",
    "terbaik",
    "membantu",
    "memuaskan",
    "puas",
    "senang",
    "suka",
    "mudah",
    "cepat",
    "lancar",
    "aman",
    "amanah",
    "terpercaya",
    "rendah",
    "kecil",
    "ringan",  # bunga rendah/kecil
    "recommended",
    "rekomendasi",
    "luar biasa",
    "terima kasih",
    "terimakasih",
    "trimakasih",
    "makasih",
    "alhamdulillah",
    "good",
    "nice",
    "excellent",
]

NEGATIVE_KW = [
    "kecewa",
    "buruk",
    "jelek",
    "parah",
    "sampah",
    "bobrok",
    "mahal",
    "tinggi",
    "besar",  # bunga mahal/tinggi
    "mencekik",
    "riba",
    "lintah",
    "teror",
    "diteror",
    "ancam",
    "diancam",
    "mengancam",
    "spam",
    "ganggu",
    "mengganggu",
    "bocor",
    "sebar",
    "disebar",
    "penipuan",
    "nipu",
    "tipu",
    "scam",
    "tidak jelas",
    "gak jelas",
    "ga jelas",
    "gajelas",
    "ditolak",
    "gagal",
    "tidak bisa",
    "menyesal",
    "nyesel",
    "rugi",
    "dirugikan",
    "kasar",
    "tidak sopan",
]


# ══════════════════════════════════════════════════════════════════════
# 1. SCRAPING
# ══════════════════════════════════════════════════════════════════════


def scrape_app(app_name: str, app_id: str, count: int) -> list[dict]:
    """Scrape reviews from Google Play Store."""
    print(f"[SCRAPE] Scraping {app_name} ({app_id}) — target {count} reviews ...")
    all_reviews = []
    batch_size = 200
    continuation_token = None

    while len(all_reviews) < count:
        result, continuation_token = reviews(
            app_id,
            lang=LANG,
            country=COUNTRY,
            sort=Sort.NEWEST,
            count=min(batch_size, count - len(all_reviews)),
            continuation_token=continuation_token,
        )
        if not result:
            break
        all_reviews.extend(result)
        print(f"  ... fetched {len(all_reviews)} reviews so far")
        if continuation_token is None:
            break

    print(f"[SCRAPE] {app_name}: scraped {len(all_reviews)} reviews total")
    return [
        {
            "app_name": app_name,
            "rating": r["score"],
            "review_text_raw": r["content"],
            "review_date": r["at"].strftime("%Y-%m-%d") if r.get("at") else "",
        }
        for r in all_reviews
        if r.get("content")
    ]


def scrape_all() -> pd.DataFrame:
    """Scrape all apps and return combined DataFrame."""
    rows = []
    for name, app_id in APPS.items():
        rows.extend(scrape_app(name, app_id, REVIEWS_PER_APP))
    df = pd.DataFrame(rows)
    print(f"\n[SCRAPE] Total raw reviews: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2b55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_newlines(text: str) -> str:
    return re.sub(r"[\r\n]+", " ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip()


def remove_unicode_artifacts(text: str) -> str:
    """Remove mathematical/styled unicode chars that mimic normal letters."""
    # Map styled unicode ranges back to ASCII
    result = []
    for ch in text:
        cp = ord(ch)
        # Mathematical bold/italic/script etc. (U+1D400 - U+1D7FF)
        if 0x1D400 <= cp <= 0x1D7FF:
            continue  # strip these styled chars
        result.append(ch)
    return "".join(result)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply preprocessing pipeline. Returns cleaned df + stats dict.
    """
    stats = {}

    # --- Raw count ---
    stats["raw_total"] = len(df)

    # --- STEP 1: Remove duplicates (by raw text) ---
    before = len(df)
    df = df.drop_duplicates(subset=["review_text_raw"], keep="first").copy()
    stats["duplicates_removed"] = before - len(df)
    stats["after_dedup"] = len(df)

    # --- STEP 2-6: Text cleaning ---
    url_count = 0
    emoji_count = 0
    newline_count = 0
    unicode_count = 0

    cleaned = []
    for raw in df["review_text_raw"]:
        text = raw

        # count & remove URLs
        urls = re.findall(r"https?://\S+|www\.\S+", text)
        url_count += len(urls)
        text = remove_urls(text)

        # count & remove emojis
        emojis_found = re.findall(
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
            text,
            flags=re.UNICODE,
        )
        emoji_count += len(emojis_found)
        text = remove_emojis(text)

        # count & remove styled unicode
        styled = [ch for ch in text if 0x1D400 <= ord(ch) <= 0x1D7FF]
        unicode_count += len(styled)
        text = remove_unicode_artifacts(text)

        # count & remove newlines
        nls = len(re.findall(r"[\r\n]", text))
        newline_count += nls
        text = remove_newlines(text)

        # lowercase
        text = text.lower()

        # normalize whitespace
        text = normalize_whitespace(text)

        cleaned.append(text)

    df["review_text"] = cleaned
    stats["urls_removed"] = url_count
    stats["emojis_removed"] = emoji_count
    stats["unicode_styled_removed"] = unicode_count
    stats["newlines_removed"] = newline_count

    # --- STEP 7: Remove empty reviews after cleaning ---
    before = len(df)
    df = df[df["review_text"].str.len() > 0].copy()
    stats["empty_removed"] = before - len(df)

    # --- STEP 8: Remove short reviews (< 3 words) ---
    before = len(df)
    df = df[df["review_text"].str.split().str.len() >= 3].copy()
    stats["short_removed"] = before - len(df)

    stats["after_cleaning"] = len(df)

    print(f"[PREP] Pipeline complete: {stats['after_cleaning']} clean reviews")
    return df, stats


# ══════════════════════════════════════════════════════════════════════
# 3. LABELING (rule-based, ±100–150 data)
# ══════════════════════════════════════════════════════════════════════


def kw_match(text: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if kw in text:
            return True
    return False


def label_aspect_sentiment(row: pd.Series) -> dict:
    """
    Return dict with risk_sentiment, trust_sentiment.
    Fill only if the aspect is clearly present.
    """
    text = row["review_text"]
    rating = row["rating"]

    is_risk = kw_match(text, RISK_KW)
    is_trust = kw_match(text, TRUST_KW)

    risk_sent = None
    trust_sent = None
    aspect_type = "None"

    if is_risk and is_trust:
        aspect_type = "Risk & Trust"
    elif is_risk:
        aspect_type = "Risk"
    elif is_trust:
        aspect_type = "Trust"

    if is_risk:
        if rating <= 2 and kw_match(text, NEGATIVE_KW):
            risk_sent = "Negative"
        elif rating >= 4 and kw_match(text, POSITIVE_KW):
            risk_sent = "Positive"
        elif rating == 3:
            risk_sent = "Neutral"
        elif rating <= 2:
            risk_sent = "Negative"
        elif rating >= 4:
            risk_sent = "Positive"

    if is_trust:
        if rating <= 2 and kw_match(text, NEGATIVE_KW):
            trust_sent = "Negative"
        elif rating >= 4 and kw_match(text, POSITIVE_KW):
            trust_sent = "Positive"
        elif rating == 3:
            trust_sent = "Neutral"
        elif rating <= 2:
            trust_sent = "Negative"
        elif rating >= 4:
            trust_sent = "Positive"

    return {
        "aspect_type": aspect_type,
        "risk_sentiment": risk_sent,
        "trust_sentiment": trust_sent,
    }


def apply_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Generate weak/noisy initial labels from rule + rating signals.
    Labels are intentionally not perfected to preserve label noise for research.
    """
    # First pass — weak labels for all rows
    labels = df.apply(label_aspect_sentiment, axis=1, result_type="expand")
    df["aspect_type"] = labels["aspect_type"]
    df["risk_sentiment"] = labels["risk_sentiment"]
    df["trust_sentiment"] = labels["trust_sentiment"]

    # A row is "labeled" if it has at least one non-null aspect sentiment
    df["_has_label"] = df["risk_sentiment"].notna() | df["trust_sentiment"].notna()

    final_labeled = df["_has_label"].sum()
    print(f"[LABEL] Weak-labeled: {final_labeled} reviews")

    # Set annotator_note to indicate weak labels
    df["annotator_note"] = ""
    df.loc[df["_has_label"], "annotator_note"] = "Weak label (rule+rating)"

    stats = {
        "auto_labeled_initial": int(final_labeled),
        "final_labeled": int(final_labeled),
        "unlabeled": int(len(df) - final_labeled),
    }

    # Count per category
    for asp in ["risk_sentiment", "trust_sentiment"]:
        for sent in ["Positive", "Negative", "Neutral"]:
            count = int((df[asp] == sent).sum())
            stats[f"{asp}_{sent}"] = count

    df.drop(columns=["_has_label"], inplace=True)

    print(f"[LABEL] Final labeled: {final_labeled}, unlabeled: {stats['unlabeled']}")
    return df, stats


# ══════════════════════════════════════════════════════════════════════
# 4. OUTPUT
# ══════════════════════════════════════════════════════════════════════


def save_csv(df: pd.DataFrame, filename: str):
    """Save final CSV with proper column order."""
    df = df.reset_index(drop=True)
    df["review_id"] = range(1, len(df) + 1)

    cols = [
        "review_id",
        "app_name",
        "rating",
        "review_date",
        "review_text",
        "aspect_type",
        "risk_sentiment",
        "trust_sentiment",
        "annotator_note",
    ]
    df = df[cols]

    # Enforce ISO date format (YYYY-MM-DD)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    df["review_date"] = df["review_date"].fillna("")

    # Replace None/NaN display
    df["aspect_type"] = df["aspect_type"].fillna("None")
    df["risk_sentiment"] = df["risk_sentiment"].fillna("")
    df["trust_sentiment"] = df["trust_sentiment"].fillna("")

    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] CSV saved: {filename} ({len(df)} rows)")
    return df


def generate_pipeline_report(
    df: pd.DataFrame,
    prep_stats: dict,
    label_stats: dict,
    filename: str,
):
    """Generate the comprehensive markdown pipeline report."""

    # ── Compute statistics ──
    total = len(df)
    kredivo_count = len(df[df["app_name"] == "Kredivo"])
    akulaku_count = len(df[df["app_name"] == "Akulaku"])

    # Rating distribution
    rating_dist = df["rating"].value_counts().sort_index()

    # Label distribution
    risk_pos = label_stats.get("risk_sentiment_Positive", 0)
    risk_neg = label_stats.get("risk_sentiment_Negative", 0)
    risk_neu = label_stats.get("risk_sentiment_Neutral", 0)
    trust_pos = label_stats.get("trust_sentiment_Positive", 0)
    trust_neg = label_stats.get("trust_sentiment_Negative", 0)
    trust_neu = label_stats.get("trust_sentiment_Neutral", 0)
    labeled = label_stats["final_labeled"]
    unlabeled = label_stats["unlabeled"]

    # Date range
    dates = pd.to_datetime(df["review_date"], errors="coerce")
    date_min = dates.min().strftime("%Y-%m-%d") if dates.notna().any() else "N/A"
    date_max = dates.max().strftime("%Y-%m-%d") if dates.notna().any() else "N/A"

    # Sample data
    def get_samples(col, val, n=3):
        subset = df[df[col] == val]
        if subset.empty:
            return []
        sample = subset.sample(min(n, len(subset)), random_state=42)
        return sample[
            ["review_id", "app_name", "rating", "review_text", col]
        ].values.tolist()

    # ── Build markdown ──
    md = []
    md.append("# Pipeline Report: Google Play Store Review Scraping & ABSA")
    md.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")

    # ── 1. Deskripsi Proses Scraping ──
    md.append("## 1. Deskripsi Proses Scraping")
    md.append("")
    md.append("| Item | Detail |")
    md.append("|------|--------|")
    md.append("| Platform | Google Play Store |")
    md.append("| Bahasa | Indonesian (`lang=id`) |")
    md.append("| Negara | Indonesia (`country=id`) |")
    md.append("| Filter Rating | 1–5 (semua) |")
    md.append("| Sort | Newest |")
    md.append(f"| Jumlah Aplikasi | {len(APPS)} |")
    md.append(f"| Target per Aplikasi | {REVIEWS_PER_APP} reviews |")
    md.append(f"| Total Raw Reviews | {prep_stats['raw_total']} |")
    md.append(f"| Periode Review | {date_min} s/d {date_max} |")
    md.append("")
    md.append("**Aplikasi yang di-scrape:**")
    md.append("")
    for name, app_id in APPS.items():
        md.append(f"- **{name}** — `{app_id}`")
    md.append("")

    # ── 2. Preprocessing Pipeline ──
    md.append("## 2. Preprocessing Pipeline")
    md.append("")
    md.append("| Step | Proses | Jumlah Item Diproses |")
    md.append("|------|--------|---------------------|")
    md.append(
        f"| 1 | Remove Duplicate (berdasarkan teks) | {prep_stats['duplicates_removed']} duplicates dihapus |"
    )
    md.append(f"| 2 | Remove URL | {prep_stats['urls_removed']} URLs dihapus |")
    md.append(f"| 3 | Remove Emoji | {prep_stats['emojis_removed']} emojis dihapus |")
    md.append(
        f"| 4 | Remove Styled Unicode | {prep_stats['unicode_styled_removed']} karakter dihapus |"
    )
    md.append(
        f"| 5 | Remove Newline/Enter | {prep_stats['newlines_removed']} newlines dihapus |"
    )
    md.append("| 6 | Lowercasing | Semua teks di-lowercase |")
    md.append("| 7 | Normalize Whitespace | Semua spasi berlebih dihapus |")
    md.append(
        f"| 8 | Remove Empty Reviews | {prep_stats['empty_removed']} reviews kosong dihapus |"
    )
    md.append(
        f"| 9 | Remove Short Reviews | {prep_stats.get('short_removed', 0)} reviews (< 3 kata) dihapus |"
    )
    md.append("")
    md.append("**Checklist Preprocessing:**")
    md.append("")
    md.append("- [x] Remove duplicate")
    md.append("- [x] Remove URL")
    md.append("- [x] Remove emoji")
    md.append("- [x] Lowercasing")
    md.append("- [x] Remove newline/enter")
    md.append("- [x] Remove short reviews (< 3 kata)")
    md.append("- [x] Preserve punctuation & slang")
    md.append("- [ ] ~~Stemming~~ (tidak dilakukan)")
    md.append("- [ ] ~~Stopword removal~~ (tidak dilakukan)")
    md.append("")
    md.append(f"**Jumlah data setelah preprocessing: {prep_stats['after_cleaning']}**")
    md.append("")

    # ── 3. Definisi Aspek ──
    md.append("## 3. Definisi Aspek")
    md.append("")
    md.append("Setiap review dievaluasi pada **dua aspek** secara independen.")
    md.append("Isi hanya jika jelas. Jika tidak jelas → kosong (NaN).")
    md.append("")
    md.append("| Aspek | Definisi | Contoh Keyword |")
    md.append("|-------|----------|----------------|")
    md.append(
        "| **Risk** (Risiko) | Bunga, denda, penagihan, keamanan data, ancaman, limit, skor kredit | bunga, denda, teror, bocor, spam, dc, blokir |"
    )
    md.append(
        "| **Trust** (Kepercayaan) | Kepercayaan ke aplikasi, legalitas, transparansi, pelayanan, kemudahan | amanah, aman, terpercaya, mudah, cepat, pelayanan, cs |"
    )
    md.append("")

    # ── 4. Labeling Guideline ──
    md.append("## 4. Labeling Guideline")
    md.append("")
    md.append("| Label | Kriteria |")
    md.append("|-------|----------|")
    md.append(
        "| **Positive** | Review menunjukkan pengalaman baik, aman, atau kepercayaan tinggi |"
    )
    md.append(
        "| **Negative** | Review menunjukkan risiko tinggi, kekecewaan, atau ketidakpercayaan |"
    )
    md.append(
        "| **Neutral** | Review informatif, ambigu, atau tidak menunjukkan emosi jelas |"
    )
    md.append("| *(kosong)* | Unlabeled — tidak dilabeli (sisa data) |")
    md.append("")
    md.append(
        "**Catatan metodologi:** Label di tahap ini adalah **label awal (weak/noisy)** dari rule + rating untuk eksperimen uncertainty-aware dan label-noise."
    )
    md.append("")
    md.append("**Skema per review:**")
    md.append("")
    md.append("```")
    md.append("review_id | aspect_type | risk_sentiment | trust_sentiment")
    md.append(
        "    1     | Risk        |   Negative     |    NaN          ← jelas risk, trust tidak jelas"
    )
    md.append(
        "    2     | Trust       |    NaN         |   Positive      ← jelas trust, risk tidak jelas"
    )
    md.append(
        "    3     | Risk & Trust|   Negative     |   Negative      ← dua aspek jelas"
    )
    md.append("    4     | None        |    NaN         |    NaN          ← unlabeled")
    md.append("```")
    md.append("")

    # ── 5. Statistik Dataset ──
    md.append("## 5. Statistik Dataset")
    md.append("")
    md.append(f"**Total data**: {total}")
    md.append(f"- Kredivo: {kredivo_count}")
    md.append(f"- Akulaku: {akulaku_count}")
    md.append("")
    md.append(f"**Data berlabel (weak labels)**: {labeled}")
    md.append(f"**Data unlabeled**: {unlabeled}")
    md.append("")

    # Rating distribution
    md.append("### Distribusi Rating")
    md.append("")
    md.append("| Rating | Jumlah | Persentase |")
    md.append("|--------|--------|------------|")
    for r in range(1, 6):
        count = int(rating_dist.get(r, 0))
        pct = f"{count / total * 100:.1f}%" if total else "0%"
        stars = "⭐" * r
        md.append(f"| {stars} ({r}) | {count} | {pct} |")
    md.append("")

    # Label distribution
    md.append("### Distribusi Label (Data Berlabel)")
    md.append("")
    md.append("| Aspek | Positive | Negative | Neutral |")
    md.append("|-------|----------|----------|---------|")
    md.append(f"| Risk | {risk_pos} | {risk_neg} | {risk_neu} |")
    md.append(f"| Trust | {trust_pos} | {trust_neg} | {trust_neu} |")
    md.append("")

    # ── 6. Contoh Data ──
    md.append("## 6. Contoh Data")
    md.append("")

    categories = [
        ("risk_sentiment", "Negative", "Risk — Negative"),
        ("risk_sentiment", "Positive", "Risk — Positive"),
        ("trust_sentiment", "Negative", "Trust — Negative"),
        ("trust_sentiment", "Positive", "Trust — Trust Positive"),
        ("risk_sentiment", "Neutral", "Neutral (Risk)"),
    ]
    for col, val, title in categories:
        samples = get_samples(col, val, 3)
        if samples:
            md.append(f"### {title}")
            md.append("")
            md.append("| ID | App | Rating | Review |")
            md.append("|----|-----|--------|--------|")
            for s in samples:
                review_short = s[3][:100] + "..." if len(s[3]) > 100 else s[3]
                review_short = review_short.replace("|", "\\|")
                md.append(f"| {s[0]} | {s[1]} | {s[2]} | {review_short} |")
            md.append("")

    # ── 7. Struktur CSV ──
    md.append("## 7. Struktur Output CSV")
    md.append("")
    md.append("| Kolom | Tipe | Deskripsi |")
    md.append("|-------|------|-----------|")
    md.append("| `review_id` | int | ID unik (1-based) |")
    md.append("| `app_name` | str | Nama aplikasi (Kredivo / Akulaku) |")
    md.append("| `rating` | int | Rating bintang (1–5) |")
    md.append("| `review_date` | str | Tanggal review (YYYY-MM-DD) |")
    md.append("| `review_text` | str | Teks review (sudah dipreprocess) |")
    md.append(
        "| `aspect_type` | str | Aspek yang dibahas (Risk / Trust / Risk & Trust / None) |"
    )
    md.append(
        "| `risk_sentiment` | str | Sentimen aspek Risk (Positive/Negative/Neutral/kosong) |"
    )
    md.append(
        "| `trust_sentiment` | str | Sentimen aspek Trust (Positive/Negative/Neutral/kosong) |"
    )
    md.append("| `annotator_note` | str | Catatan anotasi |")
    md.append("")

    # Write file
    report = "\n".join(md)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[SAVE] Markdown report saved: {filename}")
    return report


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("ABSA Pipeline — Kredivo & Akulaku Review Scraper")
    print("=" * 60)

    # Step 1: Scrape
    df_raw = scrape_all()

    # Step 2: Preprocess
    df_clean, prep_stats = preprocess(df_raw)

    # Step 3: Label
    df_labeled, label_stats = apply_labels(df_clean)

    # Step 4: Save CSV
    df_final = save_csv(df_labeled, OUTPUT_CSV)

    # Step 5: Generate report
    generate_pipeline_report(df_final, prep_stats, label_stats, OUTPUT_MD)

    print("\n" + "=" * 60)
    print("DONE! Files generated:")
    print(f"  -> {OUTPUT_CSV}")
    print(f"  -> {OUTPUT_MD}")
    print("=" * 60)


if __name__ == "__main__":
    main()
