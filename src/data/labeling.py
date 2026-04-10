"""
LLM-based silver labeling for ABSA
==================================
Annotate Indonesian fintech reviews for Risk, Trust, and Service
using an OpenAI-compatible API provider.

Usage:
    python labeling.py
    python labeling.py --batch_size 5
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import re
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from config import (
    DATA_PROCESSED,
    DATASET_ABSA_V2_PATH,
    GROQ_BATCH_SIZE,
    GROQ_MODEL,
    REVIEWS_CLEAN_V2_PATH,
    SUMOPOD_BATCH_SIZE,
    SUMOPOD_MODEL,
)

load_dotenv()

SYSTEM_PROMPT = """Kamu adalah annotator ahli untuk Aspect-Based Sentiment Analysis (ABSA) pada ulasan aplikasi fintech lending Indonesia, khususnya Kredivo dan Akulaku.
Kamu harus mengikuti Diamond Standard Guidelines secara konsisten, disiplin, dan dapat diaudit.

UNIT ANOTASI:
- Satu unit anotasi adalah satu review.
- Setiap review dinilai secara independen untuk tiga aspek: Risk, Trust, dan Service.
- Satu review dapat membahas nol, satu, atau beberapa aspek sekaligus.

DEFINISI ASPEK:
1. Risk: bunga, denda, penagihan, debt collector, ancaman, cicilan, tenor, limit kredit, skor kredit, pencairan dana, keamanan data pribadi, potongan saldo, risiko keterlambatan bayar.
2. Trust: legalitas OJK, transparansi biaya, kejujuran aplikasi, reputasi perusahaan, scam, fraud, penipuan, rasa aman terhadap institusi.
3. Service: customer service, respons keluhan, kecepatan verifikasi, login, UI/UX, bug, error teknis, kemudahan penggunaan, stabilitas aplikasi, pengalaman memakai fitur.

DEFINISI LABEL:
- Positive: pengguna secara eksplisit memuji, merasa puas, merasa terbantu, atau diuntungkan oleh aspek tersebut.
- Negative: pengguna mengeluh, marah, kecewa, merasa dirugikan, atau memberikan kritik tajam terhadap aspek tersebut.
- Neutral: pengguna membahas aspek tersebut secara faktual tanpa pujian atau keluhan yang jelas.
- null: aspek tidak dibahas sama sekali.

ATURAN INTI:
1. Fokus pada isi teks review, bukan pada rating bintang semata. Rating hanya konteks tambahan.
2. Nilai Risk, Trust, dan Service secara terpisah.
3. Jika aspek tidak dibahas, isi null. Jangan mengganti null dengan Neutral.
4. Jika satu aspek memuat pujian dan keluhan sekaligus, pilih sentimen yang paling dominan atau paling eksplisit.
5. Jika keluhan konkret lebih kuat daripada pujian umum, pilih Negative.
6. Jika review hanya berupa pertanyaan atau saran objektif tanpa emosi yang jelas, gunakan Neutral hanya untuk aspek yang benar-benar dibahas.
7. Jangan menilai benar-salah pengguna secara hukum. Fokus pada persepsi pengguna dalam teks.

ATURAN KASUS SULIT:
1. Sarkasme/ironi: gunakan makna sebenarnya, bukan kata permukaan.
2. Multi-aspek: isi semua aspek yang relevan.
3. Aspek implisit: jika nama aspek tidak disebut tetapi maknanya jelas, tetap beri label.
4. Slang/typo: pahami makna netizen Indonesia. Contoh: bapuk = jelek, scam = Trust Negative, dc galak = Risk Negative, lemot/error = Service Negative.

ATURAN REASONING:
- Reasoning wajib 1-2 kalimat singkat.
- Reasoning harus observasional, menjelaskan bukti tekstual utama.
- Jangan menulis opini pribadi atau penilaian normatif.

CHECKLIST SEBELUM MENJAWAB:
1. Semua review_id dari input harus muncul tepat satu kali di output.
2. Output harus valid JSON array.
3. Label yang valid hanya: Positive, Negative, Neutral, atau null.
4. Jangan menambah field lain selain yang diminta.
5. Jika aspek tidak dibahas, isi null.
6. Jangan menyalin rating bintang menjadi label tanpa bukti dari teks."""


SYSTEM_PROMPT_COMPACT = """Anotasi ABSA review fintech Indonesia sesuai diamond standard.
Aspek:
1. Risk = bunga, denda, penagihan, DC, ancaman, limit, cicilan, tenor, pencairan, data pribadi.
2. Trust = legalitas, transparansi, kejujuran, reputasi, scam, fraud, penipuan.
3. Service = CS, respons, verifikasi, login, UI/UX, bug, error, kemudahan, stabilitas.

Label:
- Positive = pujian/puas/terbantu
- Negative = keluhan/kecewa/dirugikan/kritik tajam
- Neutral = aspek dibahas faktual tanpa pujian/keluhan jelas
- null = aspek tidak dibahas

Aturan:
1. Nilai tiap aspek terpisah.
2. Fokus pada teks, rating hanya konteks tambahan.
3. Jangan ganti null dengan Neutral.
4. Jika campuran pujian+keluhan dalam satu aspek, pilih yang paling dominan/eksplisit.
5. Sarkasme pakai makna sebenarnya.
6. Aspek implisit tetap dilabel jika maknanya jelas.
7. Pahami slang: scam->Trust Negative, dc galak->Risk Negative, lemot/error->Service Negative.
8. Reasoning 1-2 kalimat, observasional, tanpa opini pribadi.

Output wajib valid JSON array, semua review_id muncul tepat satu kali, tanpa field tambahan."""

USER_PROMPT_TEMPLATE = """Analisis review berikut sesuai Diamond Standard Guidelines dan berikan output dalam format JSON array.

Reviews:
{reviews_json}

Output format (JSON array, satu object per review):
[
  {{
    "review_id": <id>,
    "risk_sentiment": "Positive" | "Negative" | "Neutral" | null,
    "trust_sentiment": "Positive" | "Negative" | "Neutral" | null,
    "service_sentiment": "Positive" | "Negative" | "Neutral" | null,
    "reasoning": "<penjelasan singkat>"
  }}
]

PENTING:
1. Output HANYA JSON array, tanpa markdown code blocks atau teks lain.
2. Semua review_id input harus muncul tepat satu kali.
3. Gunakan null hanya jika aspek tidak dibahas.
4. Jangan tambahkan field apa pun selain yang diminta."""


USER_PROMPT_TEMPLATE_COMPACT = """Label review berikut ke Risk/Trust/Service dengan Positive/Negative/Neutral/null.

Reviews:
{reviews_json}

Output JSON array saja:
[
    {{
        "review_id": <id>,
        "risk_sentiment": "Positive" | "Negative" | "Neutral" | null,
        "trust_sentiment": "Positive" | "Negative" | "Neutral" | null,
        "service_sentiment": "Positive" | "Negative" | "Neutral" | null,
        "reasoning": "<1-2 kalimat singkat>"
    }}
]

Semua review_id wajib muncul tepat satu kali. JSON only."""


VALID_SENTIMENTS = {"Positive", "Negative", "Neutral", None}


def recommended_groq_batch_size(model: str, prompt_mode: str) -> int | None:
    model_name = (model or "").lower()
    if "gpt-oss-20b" in model_name:
        return 10 if prompt_mode == "full" else 15
    return None


def recommended_groq_request_pause_seconds(model: str, prompt_mode: str) -> float:
    model_name = (model or "").lower()
    if "gpt-oss-20b" in model_name:
        return 3.0 if prompt_mode == "full" else 1.5
    return 0.0


def get_groq_throughput_profile(profile: str, model: str) -> dict[str, int | float | str] | None:
    model_name = (model or "").lower()
    if "gpt-oss-20b" not in model_name:
        return None

    profiles = {
        "safe": {
            "prompt_mode": "full",
            "batch_size": 10,
            "workers": 1,
            "request_pause_seconds": 3.0,
        },
        "fast": {
            "prompt_mode": "compact",
            "batch_size": 15,
            "workers": 3,
            "request_pause_seconds": 1.0,
        },
        "max": {
            "prompt_mode": "compact",
            "batch_size": 20,
            "workers": 5,
            "request_pause_seconds": 0.5,
        },
    }
    return profiles.get(profile)


def recommended_max_completion_tokens(batch_size: int, prompt_mode: str) -> int:
    # Compact mode sends denser prompts but still needs enough room for one JSON object per review.
    per_review_tokens = 220 if prompt_mode == "compact" else 260
    base_tokens = 300
    return min(8000, base_tokens + batch_size * per_review_tokens)


def extract_retry_delay_seconds(error_message: str, default_seconds: int = 60) -> int:
    match = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", error_message, re.IGNORECASE)
    if match:
        return max(default_seconds // 2, int(float(match.group(1)) + 3))
    return default_seconds


def load_existing_annotation_ids(annot_path: Path) -> set[int]:
    existing_ids = set()
    if not annot_path.exists():
        return existing_ids

    with open(annot_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "ERROR" not in str(obj.get("reasoning", "")):
                    review_id = obj.get("review_id")
                    if review_id is not None:
                        existing_ids.add(int(review_id))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    return existing_ids


def resolve_manifest_path(manifest_path: str) -> Path:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def select_reviews_from_manifest(
    reviews_df: pd.DataFrame,
    manifest_path: str,
    existing_ids: set[int],
    limit: int | None,
) -> tuple[list[dict], dict]:
    manifest_file = resolve_manifest_path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    manifest_df = pd.read_csv(manifest_file)
    if "review_id" not in manifest_df.columns:
        raise ValueError(f"Manifest must contain a review_id column: {manifest_file}")

    manifest_df = manifest_df.copy()
    manifest_df["review_id"] = pd.to_numeric(manifest_df["review_id"], errors="raise").astype("int64")

    if "manifest_order" not in manifest_df.columns:
        manifest_df.insert(0, "manifest_order", range(1, len(manifest_df) + 1))

    manifest_df = manifest_df.sort_values("manifest_order").drop_duplicates(subset=["review_id"], keep="first")

    reviews_lookup = reviews_df.copy()
    reviews_lookup["review_id"] = pd.to_numeric(reviews_lookup["review_id"], errors="raise").astype("int64")
    reviews_lookup = reviews_lookup.set_index("review_id", drop=False)

    missing_ids = [review_id for review_id in manifest_df["review_id"].tolist() if review_id not in reviews_lookup.index]
    if missing_ids:
        raise ValueError(
            f"Manifest contains review_id(s) not found in {REVIEWS_CLEAN_V2_PATH.name}: {missing_ids[:10]}"
        )

    cohort_ids = manifest_df["review_id"].tolist()
    remaining_ids = [review_id for review_id in cohort_ids if review_id not in existing_ids]
    total_remaining = len(remaining_ids)

    if limit is not None:
        remaining_ids = remaining_ids[:limit]

    pending_reviews = reviews_lookup.loc[remaining_ids].to_dict("records") if remaining_ids else []
    stats = {
        "manifest_path": str(manifest_file),
        "manifest_size": len(cohort_ids),
        "already_annotated_in_manifest": len(cohort_ids) - total_remaining,
        "remaining_in_manifest": total_remaining,
        "selected_from_manifest": len(pending_reviews),
    }
    return pending_reviews, stats


def normalize_annotation(annotation: dict, expected_review_id) -> dict:
    normalized = {
        "review_id": annotation.get("review_id", expected_review_id),
        "risk_sentiment": annotation.get("risk_sentiment"),
        "trust_sentiment": annotation.get("trust_sentiment"),
        "service_sentiment": annotation.get("service_sentiment"),
        "reasoning": str(annotation.get("reasoning", "")).strip(),
    }

    for key in ["risk_sentiment", "trust_sentiment", "service_sentiment"]:
        value = normalized.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value.lower() == "null" or value == "":
                value = None
        if value not in VALID_SENTIMENTS:
            value = None
        normalized[key] = value

    if not normalized["reasoning"]:
        normalized["reasoning"] = "No reasoning provided"

    return normalized


def validate_batch_annotations(annotations: list[dict], reviews: list[dict]) -> list[dict]:
    expected_ids = [review["review_id"] for review in reviews]
    expected_id_set = set(expected_ids)

    if not isinstance(annotations, list):
        raise ValueError("LLM output is not a list")

    normalized = []
    seen_ids = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        review_id = annotation.get("review_id")
        if review_id not in expected_id_set or review_id in seen_ids:
            continue
        normalized.append(normalize_annotation(annotation, review_id))
        seen_ids.add(review_id)

    if seen_ids != expected_id_set:
        missing_ids = [review_id for review_id in expected_ids if review_id not in seen_ids]
        raise ValueError(f"LLM output missing or mismatched review_id(s): {missing_ids[:10]}")

    ordered = sorted(normalized, key=lambda item: expected_ids.index(item["review_id"]))
    return ordered


def annotation_has_valid_label(annotation: dict) -> bool:
    return any(
        annotation.get(key) in {"Positive", "Negative", "Neutral"}
        for key in ["risk_sentiment", "trust_sentiment", "service_sentiment"]
    )


def collapse_annotations_by_review_id(annotations: list[dict]) -> pd.DataFrame:
    if not annotations:
        return pd.DataFrame()

    best_by_review_id = {}
    for index, annotation in enumerate(annotations):
        review_id = annotation.get("review_id")
        if review_id is None:
            continue
        normalized = normalize_annotation(annotation, review_id)
        normalized["_is_valid"] = annotation_has_valid_label(normalized)
        normalized["_order"] = index

        existing = best_by_review_id.get(review_id)
        if existing is None:
            best_by_review_id[review_id] = normalized
            continue

        if normalized["_is_valid"] and not existing["_is_valid"]:
            best_by_review_id[review_id] = normalized
        elif normalized["_is_valid"] == existing["_is_valid"] and normalized["_order"] > existing["_order"]:
            best_by_review_id[review_id] = normalized

    collapsed = pd.DataFrame(best_by_review_id.values())
    if collapsed.empty:
        return collapsed

    collapsed = collapsed.drop(columns=["_is_valid", "_order"], errors="ignore")
    return collapsed


def parse_llm_response(text: str) -> list[dict]:
    """Parse LLM response, handling potential markdown code blocks and conversational text."""
    text = text.strip()

    if not text:
        raise ValueError("Empty model response")
    
    # Try finding an array bracket pair if the text contains wrappers
    if not text.startswith("["):
        # Match from first [ to last ]
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)
            
    # Quick markdown strip
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:200].replace("\n", "\\n")
        raise ValueError(f"Malformed JSON response: {exc}. Preview={preview!r}") from exc


def label_batch(client, reviews: list[dict], model: str, prompt_mode: str) -> list[dict]:
    """Send a batch of reviews for annotation and validate the returned JSON."""
    compact = prompt_mode == "compact"
    reviews_json = json.dumps(
        [
            {
                "review_id": r["review_id"],
                "review_text": r["review_text"],
                "rating": r["rating"],
            }
            for r in reviews
        ],
        ensure_ascii=False,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
    )

    system_prompt = SYSTEM_PROMPT_COMPACT if compact else SYSTEM_PROMPT
    user_prompt_template = USER_PROMPT_TEMPLATE_COMPACT if compact else USER_PROMPT_TEMPLATE
    prompt = user_prompt_template.format(reviews_json=reviews_json)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_completion_tokens=recommended_max_completion_tokens(len(reviews), prompt_mode),
    )

    if response.choices[0].finish_reason == "length":
        raise ValueError(
            "Model output truncated (finish_reason=length); reduce batch_size or increase max_completion_tokens"
        )

    raw_annotations = parse_llm_response(response.choices[0].message.content)
    return validate_batch_annotations(raw_annotations, reviews)


def process_batch(client, batch_tuple, model, prompt_mode, request_pause_seconds):
    batch_num, batch = batch_tuple
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            if request_pause_seconds > 0:
                time.sleep(request_pause_seconds)

            annotations = label_batch(client, batch, model, prompt_mode)
            return batch_num, annotations, None
        except Exception as e:
            err_str = str(e).lower()
            if "401" in err_str or "invalid api key" in err_str or "authentication" in err_str:
                err_ann = []
                for r in batch:
                    err_ann.append({
                        "review_id": r["review_id"],
                        "risk_sentiment": None,
                        "trust_sentiment": None,
                        "service_sentiment": None,
                        "reasoning": f"ERROR: {e}",
                    })
                return batch_num, err_ann, str(e)
            if "429" in err_str or "quota" in err_str or "exhausted" in err_str or "retrydelay" in err_str:
                if attempt < max_retries - 1:
                    sleep_seconds = extract_retry_delay_seconds(str(e))
                    print(
                        f"  [WARN] Batch {batch_num} hit rate limit (429). "
                        f"Sleeping {sleep_seconds}s before retry {attempt+1}/{max_retries}..."
                    )
                    time.sleep(sleep_seconds)
                    continue

            if (
                "unterminated string" in err_str
                or "empty model response" in err_str
                or "malformed json response" in err_str
                or "expecting value: line 1 column 1" in err_str
            ):
                if attempt < max_retries - 1:
                    sleep_seconds = 5 + attempt * 5
                    print(
                        f"  [WARN] Batch {batch_num} returned malformed/empty JSON. "
                        f"Sleeping {sleep_seconds}s before retry {attempt+1}/{max_retries}..."
                    )
                    time.sleep(sleep_seconds)
                    continue

                recommended = recommended_groq_batch_size(model, prompt_mode)
                if recommended and len(batch) > recommended:
                    err_str = (
                        f"{e} | likely response truncation/format drift for batch_size={len(batch)}; "
                        f"try --batch_size {recommended} with model {model}"
                    )
            
            # If not a rate limit error, or we ran out of retries, fallback to empty
            err_ann = []
            for r in batch:
                err_ann.append({
                    "review_id": r["review_id"],
                    "risk_sentiment": None,
                    "trust_sentiment": None,
                    "service_sentiment": None,
                    "reasoning": f"ERROR: {e}",
                })
            return batch_num, err_ann, str(e)
            
    return batch_num, [], "Failed completely"


def select_limited_reviews(
    pending_reviews: list[dict],
    limit: int,
    strategy: str,
    seed: int,
) -> list[dict]:
    if limit >= len(pending_reviews):
        return pending_reviews

    if strategy == "head":
        return pending_reviews[:limit]

    pending_df = pd.DataFrame(pending_reviews)

    if strategy == "shuffle":
        return pending_df.sample(n=limit, random_state=seed).to_dict("records")

    if "app_name" not in pending_df.columns:
        return pending_df.sample(n=limit, random_state=seed).to_dict("records")

    if strategy == "stratified_app_rating":
        if "rating" not in pending_df.columns:
            return pending_df.sample(n=limit, random_state=seed).to_dict("records")

        strata = pending_df.groupby(["app_name", "rating"], sort=True)
        strata_df = strata.size().reset_index(name="n")
        total_pending = int(strata_df["n"].sum())
        strata_df["quota_float"] = strata_df["n"] / total_pending * limit
        strata_df["quota"] = strata_df["quota_float"].astype(int)

        remainder = limit - int(strata_df["quota"].sum())
        if remainder > 0:
            strata_df["fractional"] = strata_df["quota_float"] - strata_df["quota"]
            strata_df = strata_df.sort_values("fractional", ascending=False).reset_index(drop=True)
            strata_df.loc[: remainder - 1, "quota"] += 1
            strata_df = strata_df.sort_values(["app_name", "rating"]).reset_index(drop=True)

        sampled_parts = []
        for _, row in strata_df.iterrows():
            app_name = row["app_name"]
            rating = row["rating"]
            take_n = int(row["quota"])
            if take_n <= 0:
                continue
            group_df = pending_df[
                (pending_df["app_name"] == app_name) & (pending_df["rating"] == rating)
            ]
            sampled_parts.append(group_df.sample(n=take_n, random_state=seed))

        sampled_df = pd.concat(sampled_parts, ignore_index=True)
        sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return sampled_df.to_dict("records")

    if strategy == "equal_app":
        app_groups = list(pending_df.groupby("app_name", sort=True))
        if not app_groups:
            return []

        base_take = limit // len(app_groups)
        remainder = limit % len(app_groups)
        sampled_parts = []

        for index, (_app_name, group_df) in enumerate(app_groups):
            target_take = base_take + (1 if index < remainder else 0)
            take_n = min(len(group_df), target_take)
            sampled_parts.append(group_df.sample(n=take_n, random_state=seed))

        sampled_df = pd.concat(sampled_parts, ignore_index=True)

        if len(sampled_df) < limit:
            used_ids = set(sampled_df["review_id"].tolist())
            remaining_df = pending_df[~pending_df["review_id"].isin(used_ids)]
            extra_df = remaining_df.sample(n=limit - len(sampled_df), random_state=seed)
            sampled_df = pd.concat([sampled_df, extra_df], ignore_index=True)

        sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return sampled_df.to_dict("records")

    grouped = []
    total_pending = len(pending_df)
    app_groups = list(pending_df.groupby("app_name", sort=True))
    allocated = 0

    for index, (_app_name, group_df) in enumerate(app_groups):
        remaining_slots = limit - allocated
        remaining_apps = len(app_groups) - index
        proportional_target = round(limit * len(group_df) / total_pending)
        take_n = min(len(group_df), max(1, proportional_target))
        take_n = min(take_n, max(1, remaining_slots - (remaining_apps - 1)))
        sampled = group_df.sample(n=take_n, random_state=seed)
        grouped.append(sampled)
        allocated += take_n

    sampled_df = pd.concat(grouped, ignore_index=True)

    if len(sampled_df) < limit:
        used_ids = set(sampled_df["review_id"].tolist())
        remaining_df = pending_df[~pending_df["review_id"].isin(used_ids)]
        extra_df = remaining_df.sample(n=limit - len(sampled_df), random_state=seed)
        sampled_df = pd.concat([sampled_df, extra_df], ignore_index=True)
    elif len(sampled_df) > limit:
        sampled_df = sampled_df.sample(n=limit, random_state=seed)

    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled_df.to_dict("records")

def main():
    parser = argparse.ArgumentParser(description="Label reviews with Groq LLM")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", choices=["groq", "sumopod"], default="sumopod")
    parser.add_argument("--workers", type=int, default=1) # 1 to manage rate limits
    parser.add_argument("--limit", type=int, default=None, help="Limit total reviews to process")
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Optional CSV manifest that fixes the cohort by review_id and disables fresh sampling from the full pool",
    )
    parser.add_argument(
        "--limit_strategy",
        choices=["stratified_app_rating", "balanced_app", "equal_app", "shuffle", "head"],
        default="stratified_app_rating",
        help="How to sample reviews when --limit is used",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for limited sampling")
    parser.add_argument(
        "--prompt_mode",
        choices=["full", "compact"],
        default="full",
        help="Prompt style for labeling: full for maximum instruction fidelity, compact for lower token usage",
    )
    parser.add_argument(
        "--throughput_profile",
        choices=["safe", "fast", "max"],
        default="safe",
        help="Groq throughput preset: safe prioritizes stability, fast/max trade some stability for speed",
    )
    parser.add_argument(
        "--request_pause_seconds",
        type=float,
        default=None,
        help="Optional pause before each API call to smooth provider TPM limits",
    )
    args = parser.parse_args()
    batch_size_overridden = args.batch_size is not None
    request_pause_overridden = args.request_pause_seconds is not None
    workers_overridden = "--workers" in os.sys.argv
    prompt_mode_overridden = "--prompt_mode" in os.sys.argv

    # Set defaults based on provider if not specified
    if args.provider == "sumopod":
        if not args.model:
            args.model = SUMOPOD_MODEL
        if not args.batch_size:
            args.batch_size = SUMOPOD_BATCH_SIZE
        if args.workers == 1:
            args.workers = 15  # Maximize throughput for free tier
        api_key = os.getenv("SUMOPOD_API_KEY")
        base_url = "https://ai.sumopod.com/v1"
    else:
        if not args.model:
            args.model = GROQ_MODEL
        profile = get_groq_throughput_profile(args.throughput_profile, args.model)
        if profile:
            if not prompt_mode_overridden:
                args.prompt_mode = str(profile["prompt_mode"])
        if not args.batch_size:
            args.batch_size = int(profile["batch_size"]) if profile else GROQ_BATCH_SIZE
        if not workers_overridden and profile:
            args.workers = int(profile["workers"])
        if args.request_pause_seconds is None:
            args.request_pause_seconds = (
                float(profile["request_pause_seconds"])
                if profile
                else recommended_groq_request_pause_seconds(args.model, args.prompt_mode)
            )
        api_key = os.getenv("GROQ_API_KEY")
        base_url = "https://api.groq.com/openai/v1"

    if args.request_pause_seconds is None:
        args.request_pause_seconds = 0.0

    if not api_key:
        raise ValueError(f"{args.provider.upper()}_API_KEY not found in .env")

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
    )

    input_path = REVIEWS_CLEAN_V2_PATH
    if not input_path.exists():
        raise FileNotFoundError(f"Clean data not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"[LABEL] Loaded {len(df)} clean reviews")
    print(
        f"[LABEL] Provider={args.provider} | Model={args.model} | "
        f"Profile={args.throughput_profile} | Batch size={args.batch_size} | Workers={args.workers} | Prompt={args.prompt_mode} | "
        f"Pause={args.request_pause_seconds}s"
    )

    if args.provider == "groq":
        recommended = recommended_groq_batch_size(args.model, args.prompt_mode)
        if recommended and args.batch_size > recommended:
            source = "overridden" if batch_size_overridden else "configured"
            print(
                f"  [WARN] Groq model {args.model} with prompt={args.prompt_mode} is safer at "
                f"--batch_size {recommended}. Current {source} batch_size={args.batch_size} may cause "
                f"truncated JSON or empty responses."
            )
        recommended_pause = recommended_groq_request_pause_seconds(args.model, args.prompt_mode)
        if recommended_pause and args.request_pause_seconds < recommended_pause:
            source = "overridden" if request_pause_overridden else "configured"
            print(
                f"  [WARN] Groq model {args.model} with prompt={args.prompt_mode} is safer with "
                f"--request_pause_seconds {recommended_pause}. Current {source} pause={args.request_pause_seconds}s "
                f"may still trigger TPM rate limits on long runs."
            )

    annot_path = DATA_PROCESSED / "annotations_raw.jsonl"
    existing_ids = load_existing_annotation_ids(annot_path)
    if annot_path.exists():
        print(f"  [RESUME] Found {len(existing_ids)} existing VALID annotations")

    if args.manifest_path:
        pending_reviews, manifest_stats = select_reviews_from_manifest(
            reviews_df=df,
            manifest_path=args.manifest_path,
            existing_ids=existing_ids,
            limit=args.limit,
        )
        print(
            f"  [MANIFEST] Loaded fixed cohort from {manifest_stats['manifest_path']} "
            f"({manifest_stats['manifest_size']} review_id)"
        )
        print(
            f"  [MANIFEST] Already annotated in cohort: {manifest_stats['already_annotated_in_manifest']} | "
            f"Remaining in cohort: {manifest_stats['remaining_in_manifest']}"
        )
        if args.limit is not None:
            print(
                f"  [MANIFEST] Selected first {manifest_stats['selected_from_manifest']} remaining reviews "
                f"from fixed cohort using manifest order"
            )
    else:
        all_reviews = df.to_dict("records")
        pending_reviews = [r for r in all_reviews if r["review_id"] not in existing_ids]

        if args.limit:
            pending_reviews = select_limited_reviews(
                pending_reviews,
                limit=args.limit,
                strategy=args.limit_strategy,
                seed=args.seed,
            )
            sampled_apps = pd.DataFrame(pending_reviews)["app_name"].value_counts().to_dict()
            print(
                f"  [LIMIT] Selected {args.limit} pending reviews "
                f"using strategy={args.limit_strategy}, seed={args.seed}"
            )
            print(f"  [LIMIT] App distribution: {sampled_apps}")

    if pending_reviews:
        pending_df = pd.DataFrame(pending_reviews)
        app_distribution = {k: int(v) for k, v in pending_df["app_name"].value_counts().to_dict().items()}
        rating_distribution = {str(k): int(v) for k, v in pending_df["rating"].value_counts().sort_index().to_dict().items()}
        print(f"  Pending by app: {app_distribution}")
        print(f"  Pending by rating: {rating_distribution}")
        
    print(f"  Reviews left to process: {len(pending_reviews)}")

    if not pending_reviews:
        print("[LABEL] All reviews already annotated! Generating final CSV...")
    else:
        # Create batches
        batches = []
        for i in range(0, len(pending_reviews), args.batch_size):
            batches.append((i // args.batch_size + 1, pending_reviews[i : i + args.batch_size]))

        import concurrent.futures
        from tqdm import tqdm

        print(f"[LABEL] Starting {len(batches)} batches with {args.workers} workers...")
        print(f"[LABEL] Estimated reviews to annotate this run: {sum(len(batch) for _, batch in batches)}")
        
        # Incremental save
        saved_rows = 0
        failed_batches = 0
        error_rows = 0
        sentiment_counter = Counter()
        with open(annot_path, "a", encoding="utf-8") as f_out:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_batch,
                        client,
                        b,
                        args.model,
                        args.prompt_mode,
                        args.request_pause_seconds,
                    ): b
                    for b in batches
                }

                progress = tqdm(concurrent.futures.as_completed(futures), total=len(batches), desc="Annotating")
                for future in progress:
                    batch_num, annotations, err = future.result()
                    if err:
                        failed_batches += 1
                        tqdm.write(f"  [WARN] Batch {batch_num} failed: {err[:100]}...")
                    
                    # Save to JSONL immediately
                    for ann in annotations:
                        f_out.write(json.dumps(ann, ensure_ascii=False) + "\n")
                        f_out.flush()
                        saved_rows += 1
                        if str(ann.get("reasoning", "")).startswith("ERROR:"):
                            error_rows += 1
                        for key in ["risk_sentiment", "trust_sentiment", "service_sentiment"]:
                            value = ann.get(key)
                            if value:
                                sentiment_counter[f"{key}:{value}"] += 1

                    progress.set_postfix({
                        "rows": saved_rows,
                        "failed": failed_batches,
                    })

        print(
            f"[LABEL] Run summary: saved_rows={saved_rows}, "
            f"error_rows={error_rows}, failed_batches={failed_batches}"
        )
        if failed_batches == len(batches):
            print("[LABEL] All batches failed. Check API key, provider, or model access before trusting this run.")
        if sentiment_counter:
            top_counts = dict(sentiment_counter.most_common(6))
            print(f"[LABEL] Top sentiment counts during run: {top_counts}")

    # Re-read all from JSONL to build final CSV
    final_annotations = []
    with open(annot_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                final_annotations.append(json.loads(line))

    annot_df = collapse_annotations_by_review_id(final_annotations)

    # Normalize sentiment values
    valid_sentiments = {"Positive", "Negative", "Neutral"}
    for col in ["risk_sentiment", "trust_sentiment", "service_sentiment"]:
        if col in annot_df.columns:
            annot_df[col] = annot_df[col].apply(
                lambda x: x if x in valid_sentiments else None
            )

    # Derive aspect_type
    def get_aspect_type(row):
        aspects = []
        if pd.notna(row.get("risk_sentiment")) and row.get("risk_sentiment"):
            aspects.append("Risk")
        if pd.notna(row.get("trust_sentiment")) and row.get("trust_sentiment"):
            aspects.append("Trust")
        if pd.notna(row.get("service_sentiment")) and row.get("service_sentiment"):
            aspects.append("Service")
        return " & ".join(aspects) if aspects else "None"

    if not annot_df.empty:
        annot_df["aspect_type"] = annot_df.apply(get_aspect_type, axis=1)

    # Merge with original DataFrame
    merged = df.merge(
        annot_df[["review_id", "aspect_type", "risk_sentiment", "trust_sentiment",
                   "service_sentiment", "reasoning"]],
        on="review_id",
        how="left",
    )

    output_path = DATASET_ABSA_V2_PATH
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Print summary
    if "aspect_type" in merged.columns:
        valid_sentiments = {"Positive", "Negative", "Neutral"}
        labeled_mask = merged[["risk_sentiment", "trust_sentiment", "service_sentiment"]].isin(valid_sentiments).any(axis=1)
        labeled = merged[labeled_mask]
        print("\n[LABEL] Labeling complete!")
        print(f"  Total reviews: {len(merged)}")
        print(f"  Labeled (at least 1 aspect): {len(labeled)}")
        print(f"  Unlabeled: {len(merged) - len(labeled)}")
        for aspect in ["risk_sentiment", "trust_sentiment", "service_sentiment"]:
            if aspect in merged.columns:
                counts = merged[aspect].value_counts()
                print(f"\n  {aspect}:")
                for sent in ["Positive", "Negative", "Neutral"]:
                    print(f"    {sent}: {counts.get(sent, 0)}")

        if args.manifest_path:
            manifest_file = resolve_manifest_path(args.manifest_path)
            manifest_df = pd.read_csv(manifest_file)
            manifest_df["review_id"] = pd.to_numeric(manifest_df["review_id"], errors="raise").astype("int64")
            manifest_merged = merged[merged["review_id"].isin(manifest_df["review_id"])]
            manifest_labeled_mask = manifest_merged[
                ["risk_sentiment", "trust_sentiment", "service_sentiment"]
            ].isin(valid_sentiments).any(axis=1)
            print("\n[LABEL] Fixed cohort coverage:")
            print(f"  Cohort size: {len(manifest_df)}")
            print(f"  Cohort labeled (at least 1 aspect): {int(manifest_labeled_mask.sum())}")
            print(f"  Cohort remaining all-null/unlabeled: {int((~manifest_labeled_mask).sum())}")


if __name__ == "__main__":
    main()
