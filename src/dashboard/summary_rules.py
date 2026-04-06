"""Rule-based summary engine for the dashboard.

This module turns ABSA outputs into a short, readable summary without using an
LLM. The goal is not to generate fancy prose, but to compose consistent
sentences from clear signals in the data:

- overall sentiment distribution
- strongest positive and negative aspects
- recurring negative issue phrases
- trend signals over time
- app-level comparison when multiple apps are present

The output is a plain dictionary so the UI can render it directly as cards or
simple text blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from .aspect_taxonomy import ASPECT_DISPLAY_NAMES, ASPECT_ORDER, GENERAL_ISSUE_LABEL


VALID_SENTIMENTS: tuple[str, ...] = ("Positive", "Neutral", "Negative")
DEFAULT_APP_ORDER: tuple[str, ...] = ("Kredivo", "Akulaku")
MIN_REVIEWS_FOR_COMFORTABLE_SUMMARY = 10
MIN_REVIEWS_FOR_CONFIDENT_SUMMARY = 50

ASPECT_HINTS: dict[str, str] = {
    "risk": "limit, approval, pencairan, dan biaya",
    "trust": "kejelasan proses, privasi, dan rasa aman",
    "service": "kemudahan penggunaan dan respon bantuan",
}

ASPECT_NEGATIVE_IMPLICATIONS: dict[str, str] = {
    "risk": "Ini menunjukkan proses pembiayaan masih belum selalu terasa mulus, jelas, atau menenangkan bagi pengguna.",
    "trust": "Ini menunjukkan rasa aman dan keyakinan pengguna terhadap layanan belum sepenuhnya kuat.",
    "service": "Ini menunjukkan pengalaman penggunaan sehari-hari belum selalu terasa lancar dan konsisten.",
}

ASPECT_POSITIVE_IMPLICATIONS: dict[str, str] = {
    "risk": "Ini menunjukkan bagian yang terkait proses pembiayaan masih bisa terasa cukup terkendali bagi sebagian pengguna.",
    "trust": "Ini menunjukkan kejelasan proses dan rasa aman masih menjadi nilai yang cukup terasa.",
    "service": "Ini menunjukkan pengalaman penggunaan sehari-hari masih terasa membantu dan praktis.",
}


@dataclass(frozen=True, slots=True)
class SummaryBlock:
    """One short summary block that the UI can render as a card."""

    kind: str
    title: str
    tone: str
    text: str
    evidence: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    app_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "tone": self.tone,
            "text": self.text,
            "body": self.text,
            "evidence": list(self.evidence),
            "metrics": self.metrics,
            "app_name": self.app_name,
        }


@dataclass(frozen=True, slots=True)
class SummaryPayload:
    """Full summary payload returned by the rule engine."""

    ready: bool
    coverage: dict[str, Any]
    warnings: tuple[str, ...]
    cards: tuple[SummaryBlock, ...]
    overall: SummaryBlock
    apps: tuple[SummaryBlock, ...]
    signal: SummaryBlock
    meaning: SummaryBlock

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "status": "ready" if self.ready else "insufficient_data",
            "coverage": self.coverage,
            "warnings": list(self.warnings),
            "cards": [card.to_dict() for card in self.cards],
            "overall": self.overall.to_dict(),
            "apps": [card.to_dict() for card in self.apps],
            "focus": self.signal.to_dict(),
            "signal": self.signal.to_dict(),
            "meaning": self.meaning.to_dict(),
        }


def _pick_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _ensure_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()

    aliases = {
        "app_name": ("app_name", "app", "source_app"),
        "aspect": ("aspect",),
        "pred_label": ("pred_label", "sentiment", "label"),
        "review_date": ("review_date", "date"),
        "confidence": ("confidence", "score_confidence"),
        "review_id_ext": ("review_id_ext", "review_id", "item_id"),
        "issue": ("issue", "issue_label", "issue_text", "aspect_issue"),
        "review_text_raw": ("review_text_raw", "review_text", "text"),
    }

    for target, candidates in aliases.items():
        if target in out.columns:
            continue
        source = _pick_first_column(out, candidates)
        if source is not None:
            out[target] = out[source]

    if "review_id_ext" not in out.columns:
        out["review_id_ext"] = out.index.astype(str)

    if "review_date" in out.columns:
        out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce")

    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")

    if "aspect" in out.columns:
        out["aspect"] = out["aspect"].astype(str).str.lower().str.strip()

    if "pred_label" in out.columns:
        out["pred_label"] = out["pred_label"].astype(str).str.strip()

    if "app_name" in out.columns:
        out["app_name"] = out["app_name"].astype(str).str.strip()

    if "issue" in out.columns:
        out["issue"] = out["issue"].astype(str).str.strip()

    if "review_text_raw" in out.columns:
        out["review_text_raw"] = out["review_text_raw"].fillna("").astype(str)

    return out


def _valid_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = _ensure_canonical_columns(frame)
    if "aspect" in out.columns:
        out = out[out["aspect"].isin(ASPECT_ORDER)].copy()
    if "pred_label" in out.columns:
        out = out[out["pred_label"].isin(VALID_SENTIMENTS)].copy()
    return out.reset_index(drop=True)


def _aspect_score_table(frame: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_frame(frame)
    columns = [
        "aspect",
        "positive_share",
        "neutral_share",
        "negative_share",
        "dominant_sentiment",
        "dominant_share",
        "balance_note",
        "rows",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns)

    pivot = (
        valid.groupby(["aspect", "pred_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .pivot(index="aspect", columns="pred_label", values="count")
        .reindex(index=ASPECT_ORDER, columns=list(VALID_SENTIMENTS), fill_value=0)
        .fillna(0)
    )

    totals = pivot.sum(axis=1).replace(0, pd.NA)
    shares = pivot.div(totals, axis=0).fillna(0.0)
    dominant_sentiment = shares.idxmax(axis=1)
    dominant_share = shares.max(axis=1)

    def balance_note(positive: float, neutral: float, negative: float) -> str:
        top = max(positive, neutral, negative)
        if top < 0.45:
            return "Komposisi masih campuran"
        if negative >= positive and negative >= neutral:
            return "Negatif paling dominan"
        if positive >= negative and positive >= neutral:
            return "Positif paling dominan"
        return "Netral paling dominan"

    rows = pd.DataFrame(
        {
            "aspect": pivot.index,
            "positive_share": (shares["Positive"] * 100).round(0).astype(int),
            "neutral_share": (shares["Neutral"] * 100).round(0).astype(int),
            "negative_share": (shares["Negative"] * 100).round(0).astype(int),
            "dominant_sentiment": dominant_sentiment,
            "dominant_share": (dominant_share * 100).round(0).astype(int),
            "rows": pivot.sum(axis=1).astype(int),
        }
    ).reset_index(drop=True)
    rows["balance_note"] = [
        balance_note(float(pos), float(neu), float(neg))
        for pos, neu, neg in zip(
            shares["Positive"].tolist(),
            shares["Neutral"].tolist(),
            shares["Negative"].tolist(),
        )
    ]
    rows["aspect"] = pd.Categorical(rows["aspect"], categories=ASPECT_ORDER, ordered=True)
    return rows.sort_values("aspect").reset_index(drop=True)


def _sentiment_share_table(frame: pd.DataFrame) -> dict[str, float]:
    valid = _valid_frame(frame)
    if valid.empty:
        return {sentiment: 0.0 for sentiment in VALID_SENTIMENTS}
    shares = valid["pred_label"].value_counts(normalize=True).reindex(VALID_SENTIMENTS, fill_value=0.0)
    return {sentiment: round(float(share) * 100, 1) for sentiment, share in shares.items()}


def _dominant_sentiment(share_table: dict[str, float]) -> tuple[str, float]:
    if not share_table:
        return "Neutral", 0.0
    label = max(share_table, key=share_table.get)
    return label, float(share_table[label])


def _dominance_gap(share_table: dict[str, float], label: str) -> float:
    dominant = float(share_table.get(label, 0.0))
    others = [float(value) for key, value in share_table.items() if key != label]
    runner_up = max(others) if others else 0.0
    return dominant - runner_up


def _dominance_phrase(share_table: dict[str, float], label: str) -> str:
    dominant_share = float(share_table.get(label, 0.0))
    gap = _dominance_gap(share_table, label)

    if dominant_share < 45.0:
        return "masih campuran"
    if label == "Positive":
        if dominant_share < 55.0:
            return "masih campuran, tetapi sisi positif sedikit unggul"
        if gap < 5.0:
            return "masih campuran, tetapi sisi positif sedikit unggul"
        return "cenderung positif"
    if label == "Negative":
        if dominant_share < 55.0:
            return "masih campuran, tetapi tekanan negatif sedikit unggul"
        if gap < 5.0:
            return "masih campuran, tetapi tekanan negatif sedikit unggul"
        return "cenderung negatif"
    if gap < 5.0:
        return "masih campuran"
    return "cenderung netral"


def _dominance_tone(share_table: dict[str, float], label: str) -> str:
    dominant_share = float(share_table.get(label, 0.0))
    gap = _dominance_gap(share_table, label)
    if dominant_share >= 55.0 and gap >= 5.0:
        return label.lower()
    return "neutral"


def _app_order_from_frame(frame: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    if requested is not None:
        ordered = [str(name) for name in requested if str(name).strip()]
        if "app_name" in frame.columns:
            available = set(frame["app_name"].dropna().astype(str).tolist())
            ordered = [name for name in ordered if name in available]
        if ordered:
            return ordered

    if "app_name" not in frame.columns:
        return []

    raw = [str(name) for name in frame["app_name"].dropna().astype(str).tolist()]
    ordered = [name for name in DEFAULT_APP_ORDER if name in raw]
    ordered.extend(sorted(name for name in raw if name not in ordered))
    return ordered


def _aspect_hint(aspect: str) -> str:
    return ASPECT_HINTS.get(aspect, ASPECT_DISPLAY_NAMES.get(aspect, aspect))


def _negative_issue_sentence(aspect: str, issue_text: str | None) -> str:
    topic = issue_text or _aspect_hint(aspect)
    return f"Keluhan yang paling sering muncul terkait {topic}."


def _negative_implication_sentence(aspect: str) -> str:
    return ASPECT_NEGATIVE_IMPLICATIONS.get(
        aspect,
        "Ini menunjukkan pengalaman pengguna di area ini masih perlu perhatian lebih.",
    )


def _positive_implication_sentence(aspect: str) -> str:
    return ASPECT_POSITIVE_IMPLICATIONS.get(
        aspect,
        "Ini menunjukkan area ini masih memberi nilai yang cukup terasa bagi pengguna.",
    )


def _format_share_sentence(label: str, share: float, aspect: str | None = None) -> str:
    aspect_name = ASPECT_DISPLAY_NAMES.get(aspect, aspect.title() if aspect else "")
    if label == "Positive":
        if aspect:
            return f"Sisi yang paling sering dipuji ada pada aspek {aspect_name}, terutama {_aspect_hint(aspect)}."
        return "Sisi yang paling sering dipuji terasa pada pengalaman penggunaan."
    if label == "Negative":
        if aspect:
            return f"Tekanan terbesar ada pada aspek {aspect_name}, terutama {_aspect_hint(aspect)}."
        return "Tekanan terbesar ada pada pengalaman penggunaan secara keseluruhan."
    if aspect:
        return f"Perhatian di aspek {aspect_name} masih cukup campuran."
    return "Pengalaman pengguna masih campuran."


def _trend_sentence(frame: pd.DataFrame, aspect: str | None = None) -> str | None:
    valid = _valid_frame(frame)
    if valid.empty or "review_date" not in valid.columns:
        return None

    date_frame = valid.dropna(subset=["review_date"]).copy()
    if date_frame.empty:
        return None
    if aspect is not None:
        date_frame = date_frame[date_frame["aspect"] == aspect].copy()
        if date_frame.empty:
            return None

    daily = (
        date_frame.assign(day=date_frame["review_date"].dt.date, is_negative=date_frame["pred_label"].eq("Negative").astype(int))
        .groupby("day", as_index=False)
        .agg(neg_share=("is_negative", "mean"))
        .sort_values("day")
    )
    if len(daily) < 6:
        return None

    recent = float(daily.tail(3)["neg_share"].mean())
    previous = float(daily.tail(6).head(3)["neg_share"].mean())
    delta = (recent - previous) * 100
    if delta >= 5:
        return "Tekanan negatif cenderung naik dalam beberapa hari terakhir."
    if delta <= -5:
        return "Tekanan negatif cenderung turun dalam beberapa hari terakhir."
    return "Pergerakan sentimen relatif stabil dalam beberapa hari terakhir."


def _top_issue_text(frame: pd.DataFrame, *, aspect: str | None = None, app_name: str | None = None) -> str | None:
    valid = _valid_frame(frame)
    if valid.empty or "issue" not in valid.columns:
        return None

    subset = valid.copy()
    if aspect is not None:
        subset = subset[subset["aspect"] == aspect]
    if app_name is not None and "app_name" in subset.columns:
        subset = subset[subset["app_name"] == app_name]
    if subset.empty:
        return None

    negative = subset[subset["pred_label"] == "Negative"].copy()
    if negative.empty:
        return None

    issue_counts = (
        negative["issue"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, GENERAL_ISSUE_LABEL: pd.NA})
        .dropna()
        .value_counts()
    )
    if issue_counts.empty:
        return None

    top_issue = str(issue_counts.index[0])
    top_count = int(issue_counts.iloc[0])
    total = int(issue_counts.sum())
    if total <= 0 or top_count < 2:
        return None
    share = top_count / total * 100
    if share < 20:
        return None
    return top_issue


def _prepare_score_table(frame: pd.DataFrame, score_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if score_df is not None and not score_df.empty:
        candidate = score_df.copy()
        if "aspect" in candidate.columns:
            candidate["aspect"] = candidate["aspect"].astype(str).str.lower().str.strip()
            candidate = candidate[candidate["aspect"].isin(ASPECT_ORDER)].copy()
        required = {"positive_share", "neutral_share", "negative_share"}
        if required.issubset(candidate.columns):
            if "dominant_sentiment" not in candidate.columns:
                dominant = candidate[["positive_share", "neutral_share", "negative_share"]].idxmax(axis=1)
                candidate["dominant_sentiment"] = dominant.str.replace("_share", "", regex=False).str.title()
            if "dominant_share" not in candidate.columns:
                candidate["dominant_share"] = candidate[["positive_share", "neutral_share", "negative_share"]].max(axis=1).round(0).astype(int)
            if "balance_note" not in candidate.columns:
                candidate["balance_note"] = "Komposisi masih campuran"
            candidate["aspect"] = pd.Categorical(candidate["aspect"], categories=ASPECT_ORDER, ordered=True)
            return candidate.sort_values("aspect").reset_index(drop=True)
    return _aspect_score_table(frame)


def _summary_quality(coverage: dict[str, Any], warnings: list[str]) -> str:
    if coverage.get("unique_reviews", 0) >= MIN_REVIEWS_FOR_CONFIDENT_SUMMARY and not warnings:
        return "high"
    if coverage.get("unique_reviews", 0) >= MIN_REVIEWS_FOR_COMFORTABLE_SUMMARY:
        return "medium"
    return "low"


def _build_app_block(frame: pd.DataFrame, app_name: str) -> SummaryBlock:
    app_col = _pick_first_column(frame, ("app_name", "app", "source_app"))
    if app_col is None:
        app_frame = pd.DataFrame()
    else:
        app_frame = frame[frame[app_col].astype(str) == app_name].copy()
    if app_frame.empty:
        return SummaryBlock(
            kind="app",
            title=app_name,
            tone="neutral",
            text=f"Belum ada data yang cukup untuk membuat ringkasan {app_name}.",
            metrics={"app_name": app_name, "rows": 0},
            app_name=app_name,
        )

    share_table = _sentiment_share_table(app_frame)
    dominant_label, dominant_share = _dominant_sentiment(share_table)
    score_table = _aspect_score_table(app_frame)

    if score_table.empty:
        text = f"Iklim sentimen di {app_name} belum cukup lengkap untuk diringkas lebih jauh."
        tone = "neutral"
        metrics = {"app_name": app_name, "rows": int(len(app_frame))}
        return SummaryBlock(kind="app", title=app_name, tone=tone, text=text, metrics=metrics, app_name=app_name)

    positive_row = score_table.sort_values(["positive_share", "neutral_share"], ascending=[False, False]).iloc[0]
    negative_row = score_table.sort_values(["negative_share", "positive_share"], ascending=[False, False]).iloc[0]
    dominance_phrase = _dominance_phrase(share_table, dominant_label)

    intro = {
        "Positive": f"Di {app_name}, pengalaman pengguna {dominance_phrase} dan cukup membantu untuk kebutuhan harian.",
        "Neutral": f"Di {app_name}, pengalaman pengguna {dominance_phrase}: ada sisi yang membantu, tetapi keluhan juga masih terasa.",
        "Negative": f"Di {app_name}, pengalaman pengguna {dominance_phrase} dan lebih sering diwarnai keluhan daripada pujian.",
    }.get(dominant_label, f"Di {app_name}, pengalaman pengguna {dominance_phrase}.")

    parts = [intro]
    if positive_row["aspect"] == negative_row["aspect"]:
        aspect_name = ASPECT_DISPLAY_NAMES[str(positive_row["aspect"])]
        parts.append(f"Di sisi lain, aspek {aspect_name} paling ramai muncul baik di pujian maupun keluhan.")
    else:
        parts.append(
            f"Kekuatan paling terasa ada pada aspek {ASPECT_DISPLAY_NAMES[str(positive_row['aspect'])]}. "
        )
        parts.append(_positive_implication_sentence(str(positive_row["aspect"])))
        parts.append(f"Perhatian utama ada pada aspek {ASPECT_DISPLAY_NAMES[str(negative_row['aspect'])]}.")

    issue_text = _top_issue_text(app_frame, aspect=str(negative_row["aspect"]), app_name=app_name)
    parts.append(_negative_issue_sentence(str(negative_row["aspect"]), issue_text))
    parts.append(_negative_implication_sentence(str(negative_row["aspect"])))

    trend_text = _trend_sentence(app_frame)
    if trend_text:
        parts.append(trend_text)

    text = " ".join(parts)
    tone = "positive" if dominant_label == "Positive" else "negative" if dominant_label == "Negative" else "neutral"
    if tone != "neutral":
        tone = _dominance_tone(share_table, dominant_label)
    metrics = {
        "app_name": app_name,
        "rows": int(len(app_frame)),
        "dominant_sentiment": dominant_label,
        "dominant_share": dominant_share,
        "positive_share": share_table.get("Positive", 0.0),
        "neutral_share": share_table.get("Neutral", 0.0),
        "negative_share": share_table.get("Negative", 0.0),
        "best_aspect": str(positive_row["aspect"]),
        "worst_aspect": str(negative_row["aspect"]),
    }
    evidence = (
        f"Dominan {dominant_label} {dominant_share:.1f}%",
        f"Terkuat: {ASPECT_DISPLAY_NAMES[str(positive_row['aspect'])]}",
        f"Perlu perhatian: {ASPECT_DISPLAY_NAMES[str(negative_row['aspect'])]}",
    )
    return SummaryBlock(kind="app", title=app_name, tone=tone, text=text, evidence=evidence, metrics=metrics, app_name=app_name)


def _build_overall_block(
    frame: pd.DataFrame,
    score_table: pd.DataFrame,
    app_blocks: list[SummaryBlock],
) -> SummaryBlock:
    share_table = _sentiment_share_table(frame)
    dominant_label, dominant_share = _dominant_sentiment(share_table)

    if score_table.empty:
        return SummaryBlock(
            kind="overall",
            title="Gambaran Pengalaman",
            tone="neutral",
            text="Belum ada data yang cukup untuk membuat gambaran pengalaman secara umum.",
            metrics={"dominant_sentiment": dominant_label, "dominant_share": dominant_share},
        )

    positive_row = score_table.sort_values(["positive_share", "neutral_share"], ascending=[False, False]).iloc[0]
    negative_row = score_table.sort_values(["negative_share", "positive_share"], ascending=[False, False]).iloc[0]
    dominance_phrase = _dominance_phrase(share_table, dominant_label)

    intro = {
        "Positive": f"Secara umum, pengalaman pengguna {dominance_phrase} dan terasa membantu untuk kebutuhan harian.",
        "Neutral": f"Secara umum, pengalaman pengguna {dominance_phrase}: ada sisi yang membantu, tetapi keluhan juga masih terasa.",
        "Negative": f"Secara umum, pengalaman pengguna {dominance_phrase} dan lebih sering diwarnai keluhan daripada pujian.",
    }.get(dominant_label, f"Secara umum, pengalaman pengguna {dominance_phrase}.")

    parts = [intro]
    if positive_row["aspect"] == negative_row["aspect"]:
        aspect_name = ASPECT_DISPLAY_NAMES[str(positive_row["aspect"])]
        parts.append(f"Aspek {aspect_name} paling ramai muncul baik di pujian maupun keluhan.")
    else:
        parts.append(
            f"Sisi yang paling sering dipuji ada pada aspek {ASPECT_DISPLAY_NAMES[str(positive_row['aspect'])]}. "
        )
        parts.append(_positive_implication_sentence(str(positive_row["aspect"])))
        parts.append(f"Tekanan terbesar ada pada aspek {ASPECT_DISPLAY_NAMES[str(negative_row['aspect'])]}.")

    if len(app_blocks) >= 2:
        app_a, app_b = app_blocks[:2]
        a_neg = float(app_a.metrics.get("negative_share", 0.0))
        b_neg = float(app_b.metrics.get("negative_share", 0.0))
        a_pos = float(app_a.metrics.get("positive_share", 0.0))
        b_pos = float(app_b.metrics.get("positive_share", 0.0))
        if abs(a_neg - b_neg) >= 5 or abs(a_pos - b_pos) >= 5:
            better, worse = (app_a, app_b) if (a_pos - a_neg) >= (b_pos - b_neg) else (app_b, app_a)
            parts.append(f"Dari dua aplikasi, {better.title} terlihat lebih stabil daripada {worse.title}.")

    issue_text = _top_issue_text(frame, aspect=str(negative_row["aspect"]))
    parts.append(_negative_issue_sentence(str(negative_row["aspect"]), issue_text))
    parts.append(_negative_implication_sentence(str(negative_row["aspect"])))

    text = " ".join(parts)
    tone = "positive" if dominant_label == "Positive" else "negative" if dominant_label == "Negative" else "neutral"
    if tone != "neutral":
        tone = _dominance_tone(share_table, dominant_label)
    metrics = {
        "dominant_sentiment": dominant_label,
        "dominant_share": dominant_share,
        "best_aspect": str(positive_row["aspect"]),
        "worst_aspect": str(negative_row["aspect"]),
        "positive_share": share_table.get("Positive", 0.0),
        "neutral_share": share_table.get("Neutral", 0.0),
        "negative_share": share_table.get("Negative", 0.0),
        "unique_reviews": int(frame["review_id_ext"].nunique()) if "review_id_ext" in frame.columns else int(len(frame)),
    }
    evidence = (
        f"Dominan {dominant_label} {dominant_share:.1f}%",
        f"Terkuat: {ASPECT_DISPLAY_NAMES[str(positive_row['aspect'])]}",
        f"Perlu perhatian: {ASPECT_DISPLAY_NAMES[str(negative_row['aspect'])]}",
    )
    return SummaryBlock(
        kind="overall",
        title="Gambaran Pengalaman",
        tone=tone,
        text=text,
        evidence=evidence,
        metrics=metrics,
    )


def _build_signal_block(frame: pd.DataFrame, score_table: pd.DataFrame) -> SummaryBlock:
    if score_table.empty:
        return SummaryBlock(
            kind="signal",
            title="Sinyal yang Perlu Diperhatikan",
            tone="neutral",
            text="Belum ada sinyal yang cukup jelas untuk disorot.",
        )

    worst_row = score_table.sort_values(["negative_share", "positive_share"], ascending=[False, False]).iloc[0]
    aspect = str(worst_row["aspect"])
    issue_text = _top_issue_text(frame, aspect=aspect)
    trend_text = _trend_sentence(frame, aspect=aspect)

    parts = [
        f"Perhatian utama ada pada aspek {ASPECT_DISPLAY_NAMES[aspect]}.",
        f"Tekanan negatif paling kuat muncul di area {_aspect_hint(aspect)}.",
    ]
    parts.append(_negative_issue_sentence(aspect, issue_text))
    parts.append(_negative_implication_sentence(aspect))
    if trend_text:
        parts.append(trend_text)
    text = " ".join(parts)
    metrics = {
        "aspect": aspect,
        "negative_share": float(worst_row["negative_share"]),
        "rows": int(worst_row.get("rows", 0)),
        "issue": issue_text,
        "trend": trend_text,
    }
    evidence = (
        f"Aspect: {ASPECT_DISPLAY_NAMES[aspect]}",
        f"Negatif: {float(worst_row['negative_share']):.1f}%",
    )
    return SummaryBlock(
        kind="signal",
        title="Sinyal yang Perlu Diperhatikan",
        tone="negative",
        text=text,
        evidence=evidence,
        metrics=metrics,
    )


def _build_meaning_block(frame: pd.DataFrame, score_table: pd.DataFrame, app_blocks: list[SummaryBlock]) -> SummaryBlock:
    if score_table.empty:
        return SummaryBlock(
            kind="meaning",
            title="Makna Akhir",
            tone="neutral",
            text="Belum ada data yang cukup untuk menyimpulkan makna akhirnya.",
        )

    share_table = _sentiment_share_table(frame)
    dominant_label, _ = _dominant_sentiment(share_table)
    dominant_share = float(share_table.get(dominant_label, 0.0))
    best_row = score_table.sort_values(["positive_share", "neutral_share"], ascending=[False, False]).iloc[0]
    worst_row = score_table.sort_values(["negative_share", "positive_share"], ascending=[False, False]).iloc[0]
    dominance_phrase = _dominance_phrase(share_table, dominant_label)

    if dominant_label == "Positive":
        intro = "Artinya, aplikasi ini sudah punya fondasi pengalaman yang cukup baik."
        if dominant_share < 55.0:
            intro = "Artinya, aplikasi ini punya fondasi pengalaman yang cukup baik, tetapi belum sepenuhnya stabil."
    elif dominant_label == "Negative":
        intro = "Artinya, aplikasi ini punya nilai praktis, tetapi pengalaman belum sepenuhnya stabil."
        if dominant_share < 55.0:
            intro = "Artinya, aplikasi ini punya nilai praktis, tetapi tekanan negatifnya masih cukup terasa."
    else:
        intro = "Artinya, aplikasi ini masih berada di fase campuran."
        if dominance_phrase != "masih campuran":
            intro = f"Artinya, aplikasi ini {dominance_phrase}."

    if best_row["aspect"] == worst_row["aspect"]:
        middle = f"Hal yang paling sering dibicarakan justru ada pada aspek {ASPECT_DISPLAY_NAMES[str(best_row['aspect'])]}."
    else:
        middle = (
            f"Hal yang paling membantu ada pada aspek {ASPECT_DISPLAY_NAMES[str(best_row['aspect'])]}, "
            f"sementara titik yang paling sering memicu keluhan ada pada aspek {ASPECT_DISPLAY_NAMES[str(worst_row['aspect'])]}."
        )

    closing_parts = [intro, middle]
    if len(app_blocks) >= 2:
        app_a, app_b = app_blocks[:2]
        a_neg = float(app_a.metrics.get("negative_share", 0.0))
        b_neg = float(app_b.metrics.get("negative_share", 0.0))
        if abs(a_neg - b_neg) >= 5:
            better, worse = (app_a, app_b) if a_neg < b_neg else (app_b, app_a)
            closing_parts.append(
                f"Secara praktis, {better.title} terlihat lebih tenang, sedangkan {worse.title} masih butuh perhatian lebih."
            )

    issue_text = _top_issue_text(frame, aspect=str(worst_row["aspect"]))
    closing_parts.append(_negative_issue_sentence(str(worst_row["aspect"]), issue_text))
    closing_parts.append(_negative_implication_sentence(str(worst_row["aspect"])))
    closing_parts.append(
        f"Kalau pola ini terus berulang, kualitas pengalaman pakai dan rasa percaya terhadap layanan bisa ikut melemah."
    )

    text = " ".join(closing_parts)
    metrics = {
        "dominant_sentiment": dominant_label,
        "best_aspect": str(best_row["aspect"]),
        "worst_aspect": str(worst_row["aspect"]),
    }
    return SummaryBlock(
        kind="meaning",
        title="Makna Akhir",
        tone="neutral",
        text=text,
        metrics=metrics,
    )


def summarize_scope(
    long_df: pd.DataFrame,
    *,
    scope_name: str = "Semua aplikasi",
    score_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Summarize one ABSA scope into a compact, rule-based payload."""

    frame = _valid_frame(long_df)
    score_table = _prepare_score_table(frame, score_df)

    coverage = {
        "scope_name": scope_name,
        "rows": int(len(frame)),
        "unique_reviews": int(frame["review_id_ext"].nunique()) if "review_id_ext" in frame.columns else int(len(frame)),
        "apps": sorted(frame["app_name"].dropna().astype(str).unique().tolist()) if "app_name" in frame.columns else [],
        "has_issue_column": "issue" in frame.columns,
        "has_dates": "review_date" in frame.columns and frame["review_date"].notna().any(),
    }

    warnings: list[str] = []
    if coverage["unique_reviews"] < MIN_REVIEWS_FOR_COMFORTABLE_SUMMARY:
        warnings.append("Data masih sedikit, jadi summary dibaca sebagai indikasi awal.")
    if score_table.empty:
        warnings.append("Belum ada distribusi sentimen yang cukup untuk diringkas.")
    else:
        share_table = _sentiment_share_table(frame)
        dominant_label, dominant_share = _dominant_sentiment(share_table)
        if dominant_share < 45:
            warnings.append("Komposisi sentimen masih campuran, jadi kesimpulan dibuat hati-hati.")
        if dominant_label == "Neutral":
            warnings.append("Sentimen netral masih cukup besar.")
        if not coverage["has_issue_column"]:
            warnings.append("Tidak ada kolom issue, jadi bagian isu disederhanakan.")

    app_order = _app_order_from_frame(frame)
    app_blocks = tuple(_build_app_block(frame, app_name) for app_name in app_order)
    overall = _build_overall_block(frame, score_table, list(app_blocks))
    signal = _build_signal_block(frame, score_table)
    meaning = _build_meaning_block(frame, score_table, list(app_blocks))

    cards = (overall, *app_blocks, signal, meaning)
    ready = not frame.empty and not score_table.empty
    quality = _summary_quality(coverage, warnings)

    return SummaryPayload(
        ready=ready,
        coverage={**coverage, "quality": quality},
        warnings=tuple(warnings),
        cards=cards,
        overall=overall,
        apps=app_blocks,
        signal=signal,
        meaning=meaning,
    ).to_dict()


def summarize_app_frame(long_df: pd.DataFrame, app_name: str, score_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Summarize a single app slice using the same rule set."""

    frame = _valid_frame(long_df)
    app_col = _pick_first_column(frame, ("app_name", "app", "source_app"))
    if app_col is None:
        app_frame = pd.DataFrame()
    else:
        app_frame = frame[frame[app_col].astype(str) == app_name].copy()
    if app_frame.empty:
        empty = SummaryBlock(
            kind="app",
            title=app_name,
            tone="neutral",
            text=f"Belum ada data yang cukup untuk membuat ringkasan {app_name}.",
            metrics={"app_name": app_name, "rows": 0},
            app_name=app_name,
        )
        return empty.to_dict()
    return _build_app_block(app_frame, app_name).to_dict()


def build_summary_payload(
    long_df: pd.DataFrame,
    score_df: pd.DataFrame | None = None,
    wide_df: pd.DataFrame | None = None,
    app_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a full summary payload for the dashboard.

    ``wide_df`` is accepted for convenience and future UI hooks, but the rule
    engine mainly relies on the long ABSA frame plus an optional score table.
    """

    del wide_df  # reserved for future compatibility

    frame = _valid_frame(long_df)
    score_table = _prepare_score_table(frame, score_df)

    coverage = {
        "rows": int(len(frame)),
        "unique_reviews": int(frame["review_id_ext"].nunique()) if "review_id_ext" in frame.columns else int(len(frame)),
        "apps": sorted(frame["app_name"].dropna().astype(str).unique().tolist()) if "app_name" in frame.columns else [],
        "aspects": [aspect for aspect in ASPECT_ORDER if aspect in frame.get("aspect", pd.Series(dtype=str)).astype(str).unique().tolist()] if not frame.empty else [],
        "has_issue_column": "issue" in frame.columns,
        "has_dates": "review_date" in frame.columns and frame["review_date"].notna().any(),
    }

    warnings: list[str] = []
    if coverage["unique_reviews"] < MIN_REVIEWS_FOR_COMFORTABLE_SUMMARY:
        warnings.append("Data masih sedikit, jadi summary dibaca sebagai indikasi awal.")
    if score_table.empty:
        warnings.append("Belum ada distribusi sentimen yang cukup untuk diringkas.")
    else:
        share_table = _sentiment_share_table(frame)
        dominant_label, dominant_share = _dominant_sentiment(share_table)
        if dominant_share < 45:
            warnings.append("Komposisi sentimen masih campuran, jadi kesimpulan dibuat hati-hati.")
        if dominant_label == "Neutral":
            warnings.append("Sentimen netral masih cukup besar.")
        if not coverage["has_issue_column"]:
            warnings.append("Tidak ada kolom issue, jadi bagian isu disederhanakan.")

    selected_apps = _app_order_from_frame(frame, requested=app_names)
    if not selected_apps:
        selected_apps = _app_order_from_frame(frame)

    enough_for_app_cards = coverage["unique_reviews"] >= MIN_REVIEWS_FOR_COMFORTABLE_SUMMARY
    app_blocks = tuple(_build_app_block(frame, app_name) for app_name in selected_apps) if enough_for_app_cards else ()
    overall = _build_overall_block(frame, score_table, list(app_blocks))
    signal = _build_signal_block(frame, score_table)
    meaning = _build_meaning_block(frame, score_table, list(app_blocks))

    ready = (
        not frame.empty
        and not score_table.empty
        and coverage["unique_reviews"] >= MIN_REVIEWS_FOR_COMFORTABLE_SUMMARY
    )
    status = "ready"
    if frame.empty or score_table.empty:
        status = "no_data"
    elif coverage["unique_reviews"] < MIN_REVIEWS_FOR_COMFORTABLE_SUMMARY:
        status = "insufficient_data"

    if status == "insufficient_data":
        overall = SummaryBlock(
            kind="overall",
            title="Gambaran Pengalaman",
            tone="neutral",
            text="Data review saat ini belum cukup data untuk memberi kesimpulan yang tegas. Sinyal yang ada masih sebaiknya dibaca sebagai gambaran awal.",
            metrics={**overall.metrics, "status": status},
        )
        signal = SummaryBlock(
            kind="signal",
            title="Sinyal yang Perlu Diperhatikan",
            tone="neutral",
            text="Arah umum sudah mulai terlihat, tetapi jumlah review masih terlalu sedikit untuk menilai masalah yang paling konsisten.",
            metrics={**signal.metrics, "status": status},
        )
        meaning = SummaryBlock(
            kind="meaning",
            title="Makna Akhir",
            tone="neutral",
            text="Untuk saat ini, hasilnya lebih aman dibaca sebagai indikasi awal. Perlu tambahan review agar pola pengalaman pengguna terlihat lebih stabil.",
            metrics={**meaning.metrics, "status": status},
        )

    cards = (overall, *app_blocks, signal, meaning)
    quality = _summary_quality(coverage, warnings)

    payload = SummaryPayload(
        ready=ready,
        coverage={**coverage, "quality": quality},
        warnings=tuple(warnings),
        cards=cards,
        overall=overall,
        apps=app_blocks,
        signal=signal,
        meaning=meaning,
    )
    payload_dict = payload.to_dict()
    payload_dict["status"] = status
    return payload_dict


__all__ = [
    "SummaryBlock",
    "SummaryPayload",
    "build_summary_payload",
    "summarize_app_frame",
    "summarize_scope",
]
