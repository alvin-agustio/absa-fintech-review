from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.dashboard.aspect_taxonomy import GENERAL_ISSUE_LABEL, assign_issue_label, issue_rules


class AspectTaxonomyTests(unittest.TestCase):
    def assertIssue(self, text: str, aspect: str, expected_label: str) -> None:
        label, hits = assign_issue_label(text, aspect)
        self.assertEqual(label, expected_label, msg=f"Unexpected label for {aspect!r}: {text!r}")
        self.assertNotEqual(label, GENERAL_ISSUE_LABEL, msg=f"Should not fall back to general label for {aspect!r}: {text!r}")
        self.assertTrue(hits, msg=f"Expected at least one matched keyword for {aspect!r}: {text!r}")

    def test_trust_privacy_beats_fairness_overlap(self) -> None:
        self.assertIssue(
            "data pribadi saya disalahgunakan dan akun saya sempat diblokir",
            "trust",
            "Privasi dan penyalahgunaan data",
        )

    def test_trust_blocked_account_is_fairness_issue(self) -> None:
        self.assertIssue(
            "akun diblokir permanen tanpa alasan",
            "trust",
            "Akun, fairness, dan suspend",
        )

    def test_risk_block_after_payment_stays_on_risk(self) -> None:
        self.assertIssue(
            "setelah bayar lunas akun saya diblokir dan tidak bisa dipakai",
            "risk",
            "Blokir setelah pembayaran",
        )

    def test_risk_jatuh_tempo_is_financial_burden_not_service(self) -> None:
        self.assertIssue(
            "jatuh tempo dan telat bayar bikin denda tinggi",
            "risk",
            "Bunga, biaya, dan denda",
        )

    def test_service_specific_update_phrase_beats_generic_error(self) -> None:
        self.assertIssue(
            "error setelah update aplikasi jadi blank dan lambat",
            "service",
            "Update dan gangguan aplikasi",
        )

    def test_service_process_and_pencairan_are_grouped_together(self) -> None:
        self.assertIssue(
            "pengajuan saya sudah diproses tapi belum cair juga",
            "service",
            "Proses dan pencairan",
        )

    def test_issue_rules_expose_expected_labels(self) -> None:
        service_labels = {rule.label for rule in issue_rules("service")}
        self.assertIn("Proses dan pencairan", service_labels)
        self.assertIn("CS dan respon admin", service_labels)


if __name__ == "__main__":
    unittest.main()
