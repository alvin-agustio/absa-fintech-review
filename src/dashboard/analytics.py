"""
Analytics helpers for the Fintech Sentiment Observatory.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


ASPECT_ORDER = ["service", "trust", "risk"]
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_SCORE = {"Positive": 1, "Neutral": 0, "Negative": -1}
VALID_SENTIMENTS = set(SENTIMENT_SCORE)


def hydrate_scope(reviews_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Merge review facts with long-form predictions."""
    if reviews_df.empty or predictions_df.empty:
        return pd.DataFrame()

    reviews_df = reviews_df.copy()
    predictions_df = predictions_df.copy()
    join_keys = ["review_id_ext"]
    if "source_job_id" not in reviews_df.columns and "source_job_id" in predictions_df.columns:
        source_job_id = predictions_df["source_job_id"].iloc[0]
        reviews_df["source_job_id"] = source_job_id
    if "source_job_id" in reviews_df.columns and "source_job_id" in predictions_df.columns:
        join_keys.append("source_job_id")

    merged = predictions_df.merge(reviews_df, on=join_keys, how="left")
    merged["review_date"] = pd.to_datetime(merged["review_date"], errors="coerce")
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")
    for col in ["prob_negative", "prob_neutral", "prob_positive"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    # Unknown labels should not silently behave like neutral in downstream metrics.
    merged["pred_label"] = merged["pred_label"].where(merged["pred_label"].isin(VALID_SENTIMENTS))
    merged["sentiment_score"] = merged["pred_label"].map(SENTIMENT_SCORE)
    return merged


def wide_review_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long predictions into a per-review evidence frame."""
    if long_df.empty:
        return pd.DataFrame()

    base_cols = [
        "review_id_ext",
        "source_job_id",
        "app_id",
        "app_name",
        "review_date",
        "rating",
        "review_text_raw",
        "review_text_clean",
    ]
    wide = long_df[base_cols].drop_duplicates(subset=["review_id_ext", "source_job_id"]).copy()

    for aspect in ASPECT_ORDER:
        aspect_df = long_df[long_df["aspect"] == aspect][
            [
                "review_id_ext",
                "source_job_id",
                "pred_label",
                "confidence",
                "prob_negative",
                "prob_neutral",
                "prob_positive",
            ]
        ].rename(
            columns={
                "pred_label": f"{aspect}_sentiment",
                "confidence": f"{aspect}_confidence",
                "prob_negative": f"{aspect}_prob_negative",
                "prob_neutral": f"{aspect}_prob_neutral",
                "prob_positive": f"{aspect}_prob_positive",
            }
        )
        wide = wide.merge(aspect_df, on=["review_id_ext", "source_job_id"], how="left")

    sentiment_cols = [f"{aspect}_sentiment" for aspect in ASPECT_ORDER]
    confidence_cols = [f"{aspect}_confidence" for aspect in ASPECT_ORDER]
    wide["confidence_mean"] = wide[confidence_cols].mean(axis=1, skipna=True)
    wide["negative_votes"] = (wide[sentiment_cols] == "Negative").sum(axis=1)
    wide["positive_votes"] = (wide[sentiment_cols] == "Positive").sum(axis=1)
    wide["neutral_votes"] = (wide[sentiment_cols] == "Neutral").sum(axis=1)
    wide["mixed_signal"] = (
        ((wide[sentiment_cols] == "Negative").any(axis=1))
        & ((wide[sentiment_cols] == "Positive").any(axis=1))
    )
    wide["controversy_score"] = (
        wide["negative_votes"] * 0.5
        + wide["neutral_votes"] * 0.25
        + wide["mixed_signal"].astype(int) * 1.5
        + (1 - wide["confidence_mean"].fillna(0))
    )
    return wide.sort_values(["review_date", "review_id_ext"], ascending=[False, True]).reset_index(drop=True)


def compute_kpis(long_df: pd.DataFrame, wide_df: pd.DataFrame) -> dict[str, float | str]:
    if long_df.empty or wide_df.empty:
        return {
            "total_reviews": 0,
            "sentiment_climate": "No data",
            "aspect_pressure": "No data",
            "confidence_health": "No data",
            "avg_confidence": 0.0,
        }

    valid_long = long_df[long_df["pred_label"].isin(VALID_SENTIMENTS)].copy()
    if valid_long.empty:
        return {
            "total_reviews": int(wide_df["review_id_ext"].nunique()),
            "sentiment_climate": "No valid labels",
            "aspect_pressure": "Tidak ada",
            "confidence_health": "No data",
            "avg_confidence": 0.0,
        }

    dominant_counts = valid_long["pred_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    dominant_sentiment = dominant_counts.idxmax() if dominant_counts.sum() > 0 else "No data"

    negative_by_aspect = (
        valid_long[valid_long["pred_label"] == "Negative"]["aspect"]
        .value_counts()
        .reindex(ASPECT_ORDER, fill_value=0)
    )
    if negative_by_aspect.sum() > 0:
        aspect_pressure = negative_by_aspect.idxmax().title()
    else:
        aspect_pressure = "Tidak ada"

    avg_confidence = float(valid_long["confidence"].mean()) if valid_long["confidence"].notna().any() else 0.0
    if avg_confidence >= 0.94:
        confidence_health = "Tenang"
    elif avg_confidence >= 0.88:
        confidence_health = "Waspada"
    else:
        confidence_health = "Agresif"

    return {
        "total_reviews": int(wide_df["review_id_ext"].nunique()),
        "sentiment_climate": dominant_sentiment,
        "aspect_pressure": aspect_pressure,
        "confidence_health": confidence_health,
        "avg_confidence": avg_confidence,
    }


def sentiment_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["aspect", "pred_label", "count"])

    frame = (
        long_df.groupby(["aspect", "pred_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    frame["aspect"] = pd.Categorical(frame["aspect"], categories=ASPECT_ORDER, ordered=True)
    frame["pred_label"] = pd.Categorical(frame["pred_label"], categories=SENTIMENT_ORDER, ordered=True)
    return frame.sort_values(["aspect", "pred_label"]).reset_index(drop=True)


def trend_frame(long_df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["period", "aspect", "avg_sentiment", "avg_confidence", "count"])

    frame = long_df.copy()
    frame["period"] = frame["review_date"].dt.to_period(freq).dt.to_timestamp()
    trend = (
        frame.groupby(["period", "aspect"], as_index=False)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            avg_confidence=("confidence", "mean"),
            count=("review_id_ext", "nunique"),
        )
    )
    trend["aspect"] = pd.Categorical(trend["aspect"], categories=ASPECT_ORDER, ordered=True)
    return trend.sort_values(["period", "aspect"]).reset_index(drop=True)


def aspect_pressure_table(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["aspect", "negative_share", "avg_confidence", "volume"])

    summary = (
        long_df.assign(is_negative=long_df["pred_label"].eq("Negative").astype(int))
        .groupby("aspect", as_index=False)
        .agg(
            negative_share=("is_negative", "mean"),
            avg_confidence=("confidence", "mean"),
            volume=("review_id_ext", "nunique"),
        )
    )
    summary["negative_share"] = summary["negative_share"] * 100
    summary["aspect"] = pd.Categorical(summary["aspect"], categories=ASPECT_ORDER, ordered=True)
    return summary.sort_values("aspect").reset_index(drop=True)


def top_evidence(wide_df: pd.DataFrame, mode: str, limit: int = 8) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()

    frame = wide_df.copy()
    if mode == "negative":
        frame = frame.sort_values(["negative_votes", "confidence_mean", "review_date"], ascending=[False, False, False])
    elif mode == "controversial":
        frame = frame.sort_values(["controversy_score", "review_date"], ascending=[False, False])
    else:
        frame = frame.sort_values(["positive_votes", "confidence_mean", "review_date"], ascending=[False, False, False])
    return frame.head(limit)


def filtered_evidence(
    wide_df: pd.DataFrame,
    aspect: str | None = None,
    sentiment: str | None = None,
    app_name: str | None = None,
    min_confidence: float = 0.0,
    selected_dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()

    frame = wide_df.copy()
    if app_name and app_name != "Both":
        frame = frame[frame["app_name"] == app_name]
    if min_confidence > 0:
        frame = frame[frame["confidence_mean"].fillna(0) >= min_confidence]
    if selected_dates:
        normalized = {pd.Timestamp(d).normalize() for d in selected_dates}
        frame = frame[frame["review_date"].dt.normalize().isin(normalized)]
    if aspect:
        if sentiment:
            frame = frame[frame[f"{aspect}_sentiment"] == sentiment]
        else:
            frame = frame[frame[f"{aspect}_sentiment"].notna()]
    elif sentiment:
        mask = False
        for current_aspect in ASPECT_ORDER:
            mask = mask | frame[f"{current_aspect}_sentiment"].eq(sentiment)
        frame = frame[mask]
    return frame.reset_index(drop=True)


def review_receipt(review_row: pd.Series, long_df: pd.DataFrame) -> pd.DataFrame:
    if review_row is None or review_row.empty or long_df.empty:
        return pd.DataFrame()

    receipt = long_df[
        (long_df["review_id_ext"] == review_row["review_id_ext"])
        & (long_df["source_job_id"] == review_row["source_job_id"])
    ][
        [
            "aspect",
            "pred_label",
            "confidence",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
        ]
    ].copy()
    receipt["aspect"] = pd.Categorical(receipt["aspect"], categories=ASPECT_ORDER, ordered=True)
    return receipt.sort_values("aspect").reset_index(drop=True)


def compare_scopes(left_long: pd.DataFrame, right_long: pd.DataFrame) -> pd.DataFrame:
    if left_long.empty or right_long.empty:
        return pd.DataFrame(columns=["aspect", "metric", "left", "right", "delta"])

    def summarize(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        summary = (
            df.assign(is_negative=df["pred_label"].eq("Negative").astype(int))
            .groupby("aspect", as_index=False)
            .agg(
                negative_share=("is_negative", "mean"),
                avg_confidence=("confidence", "mean"),
                volume=("review_id_ext", "nunique"),
            )
        )
        melted = summary.melt(id_vars="aspect", var_name="metric", value_name=prefix)
        return melted

    left = summarize(left_long, "left")
    right = summarize(right_long, "right")
    merged = left.merge(right, on=["aspect", "metric"], how="outer")
    merged["delta"] = merged["right"] - merged["left"]
    merged["aspect"] = pd.Categorical(merged["aspect"], categories=ASPECT_ORDER, ordered=True)
    return merged.sort_values(["aspect", "metric"]).reset_index(drop=True)


def describe_delta(compare_df: pd.DataFrame) -> str:
    if compare_df.empty:
        return "Belum ada pembanding untuk dibaca."

    deltas = compare_df[compare_df["metric"] == "negative_share"].sort_values("delta", ascending=False)
    strongest = deltas.iloc[0]
    weakest = deltas.iloc[-1]
    return (
        f"Pressure paling naik ada di {strongest['aspect']} "
        f"({strongest['delta'] * 100:+.1f} pts), sementara perubahan paling jinak ada di "
        f"{weakest['aspect']} ({weakest['delta'] * 100:+.1f} pts)."
    )
