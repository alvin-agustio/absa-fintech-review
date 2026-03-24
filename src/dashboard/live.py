"""
Live fetch and inference orchestration for the dashboard.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Optional

import pandas as pd

from config import COUNTRY, LANG
from src.data.preprocess import preprocess_text


def build_job_key(
    app_scope: str,
    date_from: date,
    date_to: date,
    review_limit: Optional[int],
    model_id: str,
) -> str:
    limit_key = "all" if review_limit is None else str(review_limit)
    raw = f"{app_scope}|{date_from.isoformat()}|{date_to.isoformat()}|{limit_key}|{model_id}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def split_limit(total: int, n_parts: int) -> list[int]:
    base = total // n_parts
    remainder = total % n_parts
    return [base + (1 if index < remainder else 0) for index in range(n_parts)]


FETCH_AUDIT_STAGES = [
    (1, "API rows fetched"),
    (2, "Within date range + non-empty content"),
    (3, "After raw-text dedup"),
    (4, "After cleaning non-empty"),
    (5, "After minimum 3-token filter"),
]


def empty_fetch_audit_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["app_id", "app_name", "stage_order", "stage_name", "count"])


def collect_review_frames(
    app_specs: list[tuple[str, str]],
    date_from: date,
    date_to: date,
    review_limit: Optional[int],
    progress_cb=None,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    if review_limit is None:
        frames: list[pd.DataFrame] = []
        audit_frames: list[pd.DataFrame] = []
        for app_name, app_id in app_specs:
            fetched, audit_df = fetch_reviews_for_app(
                app_name,
                app_id,
                date_from,
                date_to,
                None,
                progress_cb=progress_cb,
            )
            if not fetched.empty:
                frames.append(fetched)
            if not audit_df.empty:
                audit_frames.append(audit_df)
        combined_audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else empty_fetch_audit_frame()
        return frames, combined_audit

    requested_limits: dict[str, int] = {}
    frames_by_app: dict[str, pd.DataFrame] = {}
    audit_by_app: dict[str, pd.DataFrame] = {}
    app_name_by_id = {app_id: app_name for app_name, app_id in app_specs}
    allocations = split_limit(review_limit, len(app_specs))

    for (app_name, app_id), allocation in zip(app_specs, allocations):
        if allocation <= 0:
            continue
        requested_limits[app_id] = allocation
        fetched_df, audit_df = fetch_reviews_for_app(
            app_name,
            app_id,
            date_from,
            date_to,
            allocation,
            progress_cb=progress_cb,
        )
        frames_by_app[app_id] = fetched_df
        audit_by_app[app_id] = audit_df

    remaining = max(review_limit - sum(len(frame) for frame in frames_by_app.values()), 0)
    while remaining > 0:
        expandable = [
            app_id
            for _, app_id in app_specs
            if app_id in frames_by_app
            and requested_limits.get(app_id, 0) > 0
            and len(frames_by_app[app_id]) >= requested_limits[app_id]
        ]
        if not expandable:
            break

        extra_allocations = split_limit(remaining, len(expandable))
        progress_made = False
        for app_id, extra in zip(expandable, extra_allocations):
            if extra <= 0:
                continue
            new_target = requested_limits[app_id] + extra
            refetched, audit_df = fetch_reviews_for_app(
                app_name_by_id[app_id],
                app_id,
                date_from,
                date_to,
                new_target,
                progress_cb=progress_cb,
            )
            if len(refetched) > len(frames_by_app[app_id]):
                progress_made = True
            frames_by_app[app_id] = refetched
            audit_by_app[app_id] = audit_df
            requested_limits[app_id] = new_target

        new_remaining = max(review_limit - sum(len(frame) for frame in frames_by_app.values()), 0)
        if new_remaining >= remaining and not progress_made:
            break
        remaining = new_remaining

    ordered_frames: list[pd.DataFrame] = []
    for _, app_id in app_specs:
        frame = frames_by_app.get(app_id)
        if frame is not None and not frame.empty:
            ordered_frames.append(frame)
    combined_audit = pd.concat(list(audit_by_app.values()), ignore_index=True) if audit_by_app else empty_fetch_audit_frame()
    return ordered_frames, combined_audit


def fetch_reviews_for_app(
    app_name: str,
    app_id: str,
    date_from: date,
    date_to: date,
    target_count: Optional[int],
    progress_cb=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from google_play_scraper import Sort, reviews
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dependency live fetch belum tersedia di environment ini: `google_play_scraper`. "
            "Install paket `google-play-scraper` pada Python environment yang dipakai untuk menjalankan Streamlit."
        ) from exc

    candidate_reviews: list[dict] = []
    total_api_rows = 0
    continuation_token = None
    batch_size = 200

    while True:
        if target_count is not None and len(candidate_reviews) >= target_count:
            break

        request_count = batch_size
        if target_count is not None:
            request_count = min(batch_size, max(target_count - len(candidate_reviews), 1))

        result, continuation_token = reviews(
            app_id,
            lang=LANG,
            country=COUNTRY,
            sort=Sort.NEWEST,
            count=request_count,
            continuation_token=continuation_token,
        )
        if not result:
            break
        total_api_rows += len(result)

        oldest_in_batch = None

        for item in result:
            posted = item.get("at")
            if not posted:
                continue
            posted_date = posted.date()
            oldest_in_batch = posted_date if oldest_in_batch is None else min(oldest_in_batch, posted_date)
            if date_from <= posted_date <= date_to and item.get("content"):
                candidate_reviews.append(
                    {
                        "review_id_ext": f"{app_id}-{item.get('reviewId', len(candidate_reviews))}",
                        "app_id": app_id,
                        "app_name": app_name,
                        "rating": item.get("score"),
                        "review_date": posted_date.isoformat(),
                        "review_text_raw": item.get("content", "").strip(),
                    }
                )
        if progress_cb:
            progress_total = target_count if target_count is not None else max(len(candidate_reviews), 1)
            progress_cb("fetching", app_name, len(candidate_reviews), progress_total)

        # Results are sorted newest-first. If oldest item in this page is already
        # older than date_from, all following pages are older and can be skipped.
        if target_count is None and oldest_in_batch is not None and oldest_in_batch < date_from:
            break

        if continuation_token is None:
            break

    base_columns = [
        "review_id_ext",
        "app_id",
        "app_name",
        "rating",
        "review_date",
        "review_text_raw",
        "review_text_clean",
    ]
    if not candidate_reviews:
        audit_rows = [
            {
                "app_id": app_id,
                "app_name": app_name,
                "stage_order": stage_order,
                "stage_name": stage_name,
                "count": total_api_rows if stage_order == 1 else 0,
            }
            for stage_order, stage_name in FETCH_AUDIT_STAGES
        ]
        return pd.DataFrame(columns=base_columns), pd.DataFrame(audit_rows)

    scoped_df = pd.DataFrame(candidate_reviews)
    dedup_df = scoped_df.drop_duplicates(subset=["review_text_raw"]).copy()
    clean_df = dedup_df.copy()
    clean_df["review_text_clean"] = clean_df["review_text_raw"].astype(str).map(preprocess_text)
    clean_nonempty_df = clean_df[clean_df["review_text_clean"].str.len() > 0].copy()
    final_df = clean_nonempty_df[clean_nonempty_df["review_text_clean"].str.split().str.len() >= 3].reset_index(drop=True)

    audit_counts = {
        1: int(total_api_rows),
        2: int(len(scoped_df)),
        3: int(len(dedup_df)),
        4: int(len(clean_nonempty_df)),
        5: int(len(final_df)),
    }
    audit_rows = [
        {
            "app_id": app_id,
            "app_name": app_name,
            "stage_order": stage_order,
            "stage_name": stage_name,
            "count": audit_counts.get(stage_order, 0),
        }
        for stage_order, stage_name in FETCH_AUDIT_STAGES
    ]

    if target_count is None:
        return final_df, pd.DataFrame(audit_rows)
    return final_df.head(target_count), pd.DataFrame(audit_rows)


def build_predictions_fact(
    reviews_df: pd.DataFrame,
    results: list[dict],
    model_id: str,
    job_id: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, result in enumerate(results):
        review_row = reviews_df.iloc[idx]
        for aspect, pred in result.items():
            if aspect == "review_text" or pred is None:
                continue
            rows.append(
                {
                    "review_id_ext": review_row["review_id_ext"],
                    "source_job_id": job_id,
                    "model_id": model_id,
                    "aspect": aspect,
                    "pred_label": pred["sentiment"],
                    "confidence": pred["confidence"],
                    "prob_negative": pred["prob_negative"],
                    "prob_neutral": pred["prob_neutral"],
                    "prob_positive": pred["prob_positive"],
                }
            )
    return pd.DataFrame(rows)


def run_live_analysis(
    store,
    model_id: str,
    app_specs: list[tuple[str, str]],
    date_from: date,
    date_to: date,
    review_limit: Optional[int],
    predictor=None,
    predictor_factory=None,
    progress_cb=None,
    allow_cached: bool = True,
):
    app_scope = ",".join(app_id for _, app_id in app_specs)
    job_key = build_job_key(app_scope, date_from, date_to, review_limit, model_id)
    cached_job = store.find_cached_job(job_key) if allow_cached else None
    if cached_job:
        cached_job_id = cached_job["job_id"] if isinstance(cached_job, dict) else str(cached_job)
        reviews_df, predictions_df = store.load_job_frames(cached_job_id)
        fetch_audit_df = store.load_live_fetch_audit(cached_job_id)
        return {
            "job_id": cached_job_id,
            "job_key": job_key,
            "cached": True,
            "reviews_df": reviews_df,
            "predictions_df": predictions_df,
            "fetch_audit_df": fetch_audit_df,
        }

    review_frames, fetch_audit_df = collect_review_frames(
        app_specs=app_specs,
        date_from=date_from,
        date_to=date_to,
        review_limit=review_limit,
        progress_cb=progress_cb,
    )

    reviews_df = (
        pd.concat(review_frames, ignore_index=True)
        if review_frames
        else pd.DataFrame(columns=["review_id_ext", "app_id", "app_name", "rating", "review_date", "review_text_raw", "review_text_clean"])
    )
    job_id = job_key[:12]
    job_meta = {
        "job_id": job_id,
        "job_key": job_key,
        "app_id": app_scope,
        "app_name": "Both" if len(app_specs) > 1 else app_specs[0][0],
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "review_limit": review_limit,
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
        "model_id": model_id,
        "status": "completed",
    }
    if reviews_df.empty:
        store.save_live_job(job_meta, reviews_df, pd.DataFrame(), fetch_audit_df=fetch_audit_df)
        return {
            "job_id": job_id,
            "job_key": job_key,
            "cached": False,
            "reviews_df": reviews_df,
            "predictions_df": pd.DataFrame(),
            "fetch_audit_df": fetch_audit_df,
        }

    if predictor is None:
        if predictor_factory is None:
            raise ValueError("predictor or predictor_factory must be provided for uncached live analysis")
        predictor = predictor_factory()

    if progress_cb:
        progress_cb("inferencing", "all", len(reviews_df), len(reviews_df))
    results = predictor.predict(reviews_df["review_text_clean"].tolist())

    reviews_df = reviews_df.copy()
    reviews_df["source_job_id"] = job_id
    predictions_df = build_predictions_fact(reviews_df, results, model_id=model_id, job_id=job_id)
    store.save_live_job(job_meta, reviews_df, predictions_df, fetch_audit_df=fetch_audit_df)
    return {
        "job_id": job_id,
        "job_key": job_key,
        "cached": False,
        "reviews_df": reviews_df,
        "predictions_df": predictions_df,
        "fetch_audit_df": fetch_audit_df,
    }
