from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.audit_summary_rules import build_audit_report, render_markdown  # noqa: E402


def _review(app_name: str, aspect: str, sentiment: str, idx: int) -> dict[str, object]:
    return {
        "review_id_ext": f"{app_name.lower()}-{aspect}-{idx}",
        "source_job_id": "job-1",
        "app_name": app_name,
        "aspect": aspect,
        "pred_label": sentiment,
        "review_date": pd.Timestamp("2026-03-01") + pd.Timedelta(days=idx),
        "review_text_raw": f"{app_name} {aspect} {sentiment} review {idx}",
    }


class SummaryAuditTests(unittest.TestCase):
    def test_audit_report_flags_app_contrast_and_missing_issue_column(self) -> None:
        long_df = pd.DataFrame(
            [
                _review("Kredivo", "risk", "Positive", 0),
                _review("Kredivo", "risk", "Positive", 1),
                _review("Kredivo", "trust", "Positive", 2),
                _review("Kredivo", "service", "Positive", 3),
                _review("Akulaku", "risk", "Negative", 4),
                _review("Akulaku", "trust", "Negative", 5),
                _review("Akulaku", "service", "Negative", 6),
                _review("Akulaku", "service", "Neutral", 7),
                _review("Akulaku", "risk", "Negative", 8),
                _review("Kredivo", "trust", "Positive", 9),
            ]
        )

        report = build_audit_report(long_df, source_label="synthetic")
        guardrails = {item["name"]: item for item in report["guardrails"]}

        self.assertEqual(report["status"], "ready")
        self.assertEqual(guardrails["both_apps_present"]["status"], "pass")
        self.assertEqual(guardrails["app_order_is_kredivo_first"]["status"], "pass")
        self.assertEqual(guardrails["app_contrast_visible"]["status"], "pass")
        self.assertEqual(guardrails["issue_column_available"]["status"], "info")
        self.assertIn("Kredivo", report["summary"]["overall"]["body"])
        self.assertIn("Akulaku", report["summary"]["meaning"]["body"])

        markdown = render_markdown(report)
        self.assertIn("# Summary Rules Audit", markdown)
        self.assertIn("Per-App Snapshot", markdown)
        self.assertIn("Kredivo", markdown)
        self.assertIn("Akulaku", markdown)

    def test_audit_report_marks_single_app_as_no_contrast(self) -> None:
        long_df = pd.DataFrame(
            [
                _review("Kredivo", "risk", "Positive", 0),
                _review("Kredivo", "trust", "Positive", 1),
                _review("Kredivo", "service", "Neutral", 2),
                _review("Kredivo", "service", "Positive", 3),
                _review("Kredivo", "risk", "Negative", 4),
                _review("Kredivo", "trust", "Positive", 5),
                _review("Kredivo", "service", "Positive", 6),
                _review("Kredivo", "service", "Positive", 7),
                _review("Kredivo", "risk", "Positive", 8),
                _review("Kredivo", "trust", "Neutral", 9),
            ]
        )

        report = build_audit_report(long_df, source_label="single-app")
        guardrails = {item["name"]: item for item in report["guardrails"]}

        self.assertEqual(guardrails["both_apps_present"]["status"], "warn")
        self.assertEqual(guardrails["app_contrast_visible"]["status"], "warn")
        self.assertEqual(report["summary"]["apps"][0]["app_name"], "Kredivo")


if __name__ == "__main__":
    unittest.main()
