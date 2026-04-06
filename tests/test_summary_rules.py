from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.dashboard import summary_rules


def _review(app_name: str, aspect: str, sentiment: str, idx: int) -> dict[str, object]:
    return {
        "review_id_ext": f"{app_name.lower()}-{aspect}-{idx}",
        "app_name": app_name,
        "aspect": aspect,
        "pred_label": sentiment,
        "review_date": pd.Timestamp("2026-03-01") + pd.Timedelta(days=idx),
        "review_text_raw": f"{app_name} {aspect} {sentiment} review {idx}",
    }


def _score_row(
    aspect: str,
    positive: int,
    neutral: int,
    negative: int,
) -> dict[str, object]:
    shares = {"Positive": positive, "Neutral": neutral, "Negative": negative}
    dominant_sentiment = max(shares, key=shares.get)
    return {
        "aspect": aspect,
        "positive_share": positive,
        "neutral_share": neutral,
        "negative_share": negative,
        "dominant_sentiment": dominant_sentiment,
        "dominant_share": shares[dominant_sentiment],
        "balance_note": f"{dominant_sentiment} paling dominan",
    }


def _build_payload(long_rows: list[dict[str, object]], score_rows: list[dict[str, object]]) -> dict[str, object]:
    long_df = pd.DataFrame(long_rows)
    score_df = pd.DataFrame(score_rows)
    return summary_rules.build_summary_payload(long_df, score_df)


class SummaryRulesTests(unittest.TestCase):
    def test_insufficient_data_returns_cautious_summary(self) -> None:
        payload = _build_payload(
            [
                _review("Kredivo", "risk", "Negative", 0),
                _review("Akulaku", "trust", "Neutral", 1),
                _review("Kredivo", "service", "Positive", 2),
            ],
            [
                _score_row("risk", 20, 10, 70),
                _score_row("trust", 30, 40, 30),
                _score_row("service", 60, 20, 20),
            ],
        )

        self.assertEqual(payload.get("status"), "insufficient_data")
        self.assertIn("belum cukup data", str(payload.get("overall", {}).get("body", "")).lower())
        self.assertEqual(payload.get("apps"), [])

    def test_dominant_sentiment_shapes_overall_and_focus_summary(self) -> None:
        payload = _build_payload(
            [
                _review("Kredivo", "risk", "Negative", 0),
                _review("Kredivo", "trust", "Negative", 1),
                _review("Kredivo", "service", "Negative", 2),
                _review("Akulaku", "risk", "Negative", 3),
                _review("Akulaku", "trust", "Positive", 4),
                _review("Akulaku", "service", "Negative", 5),
                _review("Akulaku", "service", "Neutral", 6),
                _review("Kredivo", "risk", "Positive", 7),
                _review("Akulaku", "trust", "Negative", 8),
                _review("Kredivo", "service", "Negative", 9),
            ],
            [
                _score_row("risk", 20, 10, 70),
                _score_row("trust", 30, 15, 55),
                _score_row("service", 35, 10, 55),
            ],
        )

        self.assertEqual(payload.get("status"), "ready")
        overall_body = str(payload.get("overall", {}).get("body", "")).lower()
        focus_body = str(payload.get("focus", {}).get("body", "")).lower()

        self.assertIn("negatif", overall_body)
        self.assertIn("risk", overall_body)
        self.assertIn("risk", focus_body)

    def test_app_summaries_distinguish_kredivo_and_akulaku(self) -> None:
        payload = _build_payload(
            [
                _review("Kredivo", "service", "Positive", 0),
                _review("Kredivo", "service", "Positive", 1),
                _review("Kredivo", "trust", "Positive", 2),
                _review("Kredivo", "risk", "Neutral", 3),
                _review("Kredivo", "service", "Positive", 4),
                _review("Akulaku", "risk", "Negative", 5),
                _review("Akulaku", "risk", "Negative", 6),
                _review("Akulaku", "trust", "Negative", 7),
                _review("Akulaku", "service", "Negative", 8),
                _review("Akulaku", "service", "Neutral", 9),
            ],
            [
                _score_row("risk", 15, 10, 75),
                _score_row("trust", 35, 10, 55),
                _score_row("service", 55, 10, 35),
            ],
        )

        apps = payload.get("apps", [])
        self.assertGreaterEqual(len(apps), 2)
        self.assertEqual(apps[0].get("app_name"), "Kredivo")
        self.assertEqual(apps[1].get("app_name"), "Akulaku")
        self.assertIn("positif", str(apps[0].get("body", "")).lower())
        self.assertIn("negatif", str(apps[1].get("body", "")).lower())

    def test_near_tie_app_summary_softens_to_neutral(self) -> None:
        payload = _build_payload(
            [
                _review("Kredivo", "risk", "Positive", 0),
                _review("Kredivo", "trust", "Positive", 1),
                _review("Kredivo", "service", "Positive", 2),
                _review("Kredivo", "risk", "Positive", 3),
                _review("Kredivo", "trust", "Negative", 4),
                _review("Kredivo", "service", "Negative", 5),
                _review("Kredivo", "risk", "Negative", 6),
                _review("Kredivo", "trust", "Positive", 7),
                _review("Kredivo", "service", "Negative", 8),
                _review("Kredivo", "risk", "Neutral", 9),
            ],
            [
                _score_row("risk", 50, 10, 40),
                _score_row("trust", 50, 10, 40),
                _score_row("service", 40, 10, 50),
            ],
        )

        self.assertGreaterEqual(len(payload.get("apps", [])), 1)
        app = payload["apps"][0]
        self.assertEqual(app.get("app_name"), "Kredivo")
        self.assertEqual(app.get("tone"), "neutral")
        self.assertIn("masih campuran", str(app.get("body", "")).lower())


if __name__ == "__main__":
    unittest.main()
