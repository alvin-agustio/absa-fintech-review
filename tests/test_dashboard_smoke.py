from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.dashboard.live import run_live_analysis
from src.dashboard.summary_rules import build_summary_payload


class _FakeStore:
    def __init__(self) -> None:
        self.saved_meta = None
        self.saved_reviews = None
        self.saved_predictions = None
        self.saved_fetch_audit = None

    def find_cached_job(self, job_key: str):
        return None

    def save_live_job(self, meta, reviews_df, predictions_df, fetch_audit_df=None) -> None:
        self.saved_meta = meta
        self.saved_reviews = reviews_df.copy()
        self.saved_predictions = predictions_df.copy()
        self.saved_fetch_audit = fetch_audit_df.copy() if fetch_audit_df is not None else None


class _FakePredictor:
    def predict(self, texts: list[str]) -> list[dict]:
        results = []
        for text in texts:
            del text
            results.append(
                {
                    "review_text": "ignored",
                    "risk": {
                        "sentiment": "Negative",
                        "confidence": 0.91,
                        "prob_negative": 0.91,
                        "prob_neutral": 0.05,
                        "prob_positive": 0.04,
                    },
                    "trust": {
                        "sentiment": "Neutral",
                        "confidence": 0.62,
                        "prob_negative": 0.17,
                        "prob_neutral": 0.62,
                        "prob_positive": 0.21,
                    },
                    "service": {
                        "sentiment": "Positive",
                        "confidence": 0.73,
                        "prob_negative": 0.10,
                        "prob_neutral": 0.17,
                        "prob_positive": 0.73,
                    },
                }
            )
        return results


class DashboardSmokeTests(unittest.TestCase):
    def test_build_summary_payload_accepts_alias_columns_and_requested_app_order(self) -> None:
        rows = []
        for idx in range(6):
            rows.append(
                {
                    "app": "Kredivo",
                    "aspect": "service",
                    "sentiment": "Positive",
                    "date": f"2026-03-{idx + 1:02d}",
                    "issue_label": "app cepat",
                    "text": f"kredivo review {idx}",
                }
            )
        for idx in range(6, 12):
            rows.append(
                {
                    "app": "Akulaku",
                    "aspect": "risk",
                    "sentiment": "Negative",
                    "date": f"2026-03-{idx + 1:02d}",
                    "issue_label": "biaya tinggi",
                    "text": f"akulaku review {idx}",
                }
            )

        payload = build_summary_payload(pd.DataFrame(rows), app_names=["Akulaku", "Kredivo"])

        self.assertEqual(payload["status"], "ready")
        self.assertEqual([card["app_name"] for card in payload["apps"]], ["Akulaku", "Kredivo"])
        self.assertTrue(payload["coverage"]["has_issue_column"])
        self.assertTrue(payload["coverage"]["has_dates"])

    def test_run_live_analysis_runs_uncached_flow_without_network_or_weights(self) -> None:
        fetched_reviews = pd.DataFrame(
            [
                {
                    "review_id_ext": "review-1",
                    "app_id": "com.finaccel.android",
                    "app_name": "Kredivo",
                    "rating": 1,
                    "review_date": "2026-03-01",
                    "review_text_raw": "tagihan bikin bingung",
                    "review_text_clean": "tagihan bikin bingung",
                },
                {
                    "review_id_ext": "review-2",
                    "app_id": "com.finaccel.android",
                    "app_name": "Kredivo",
                    "rating": 4,
                    "review_date": "2026-03-02",
                    "review_text_raw": "aplikasi lumayan membantu",
                    "review_text_clean": "aplikasi lumayan membantu",
                },
            ]
        )
        fetch_audit = pd.DataFrame(
            [
                {
                    "app_id": "com.finaccel.android",
                    "app_name": "Kredivo",
                    "stage_order": 1,
                    "stage_name": "API rows fetched",
                    "count": 2,
                }
            ]
        )
        store = _FakeStore()
        predictor = _FakePredictor()

        with patch(
            "src.dashboard.live.collect_review_frames",
            return_value=([fetched_reviews], fetch_audit),
        ):
            result = run_live_analysis(
                store=store,
                model_id="baseline_epoch_5",
                app_specs=[("Kredivo", "com.finaccel.android")],
                date_from=date(2026, 3, 1),
                date_to=date(2026, 3, 31),
                review_limit=2,
                predictor=predictor,
                allow_cached=False,
            )

        self.assertFalse(result["cached"])
        self.assertEqual(len(result["reviews_df"]), 2)
        self.assertEqual(len(result["predictions_df"]), 6)
        self.assertEqual(store.saved_meta["model_id"], "baseline_epoch_5")
        self.assertTrue((result["predictions_df"]["source_job_id"] == result["job_id"]).all())
        self.assertEqual(len(result["fetch_audit_df"]), 1)


if __name__ == "__main__":
    unittest.main()
