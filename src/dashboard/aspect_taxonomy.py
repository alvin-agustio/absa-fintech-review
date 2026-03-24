"""Reusable aspect taxonomy and lightweight presence detection for the dashboard.

This module centralizes the canonical dashboard vocabulary for the three ABSA
aspects used throughout the project:

- ``risk``
- ``trust``
- ``service``

It provides two small layers of helpers:

1. A stronger issue taxonomy for human-readable issue labels and keywords.
2. A lightweight keyword-based aspect presence detector.

The API is intentionally small so ``app.py`` or other dashboard modules can
import only what they need without depending on UI code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


ASPECT_ORDER: tuple[str, ...] = ("risk", "trust", "service")
ASPECT_DISPLAY_NAMES: dict[str, str] = {
    "risk": "Risk",
    "trust": "Trust",
    "service": "Service",
}

GENERAL_ISSUE_LABEL = "Belum cukup spesifik"


@dataclass(frozen=True, slots=True)
class IssueRule:
    """A human-readable issue bucket and its keyword hints."""

    label: str
    keywords: tuple[str, ...]


# Stronger, broader taxonomy than the inline dashboard rules.
# The labels are meant to be readable in cards and diagnostics.
ISSUE_TAXONOMY: dict[str, tuple[IssueRule, ...]] = {
    "risk": (
        IssueRule(
            label="Limit, approval, dan pencairan",
            keywords=(
                "limit",
                "limit naik",
                "limit turun",
                "ditolak",
                "penolakan",
                "tolak",
                "pengajuan",
                "approve",
                "acc",
                "cair",
                "pencairan",
                "dana cair",
                "pinjaman",
            ),
        ),
        IssueRule(
            label="Bunga, biaya, dan denda",
            keywords=(
                "bunga",
                "biaya",
                "biaya admin",
                "admin",
                "denda",
                "tagihan",
                "cicilan",
                "tenor",
                "pelunasan",
                "mahal",
                "riba",
                "mencekik",
                "potongan",
            ),
        ),
        IssueRule(
            label="Penagihan dan debt collector",
            keywords=(
                "penagihan",
                "tagih",
                "dc",
                "debt collector",
                "debt",
                "telepon",
                "ancam",
                "ancaman",
                "teror",
                "kasar",
                "sebar data",
                "data sebar",
            ),
        ),
        IssueRule(
            label="Blokir setelah pembayaran",
            keywords=(
                "blokir",
                "diblokir",
                "beku",
                "dibekukan",
                "suspend",
                "freeze",
                "pelunasan",
                "lunas",
                "bayar cepat",
                "setelah bayar",
            ),
        ),
        IssueRule(
            label="Keamanan data pribadi",
            keywords=(
                "data",
                "privasi",
                "keamanan",
                "kontak",
                "nomor",
                "izin",
                "akses",
                "bocor",
                "sebar",
                "disalahgunakan",
                "penyalahgunaan",
            ),
        ),
    ),
    "trust": (
        IssueRule(
            label="Transparansi dan kejelasan",
            keywords=(
                "tidak jelas",
                "ga jelas",
                "gak jelas",
                "tanpa alasan",
                "alasan",
                "transparan",
                "transparansi",
                "kejelasan",
                "sepihak",
                "status",
                "informasi",
                "pemberitahuan",
                "notifikasi",
                "konfirmasi",
                "janji",
                "kepastian",
            ),
        ),
        IssueRule(
            label="Legalitas, OJK, dan reputasi",
            keywords=(
                "legal",
                "legalitas",
                "ojk",
                "resmi",
                "terpercaya",
                "aman",
                "amanah",
                "izin",
                "berizin",
                "reputasi",
                "perusahaan",
            ),
        ),
        IssueRule(
            label="Penipuan dan fraud",
            keywords=(
                "tipu",
                "penipu",
                "penipuan",
                "scam",
                "hoax",
                "bohong",
                "palsu",
                "fraud",
                "abal abal",
                "ilegal",
            ),
        ),
        IssueRule(
            label="Akun, fairness, dan suspend",
            keywords=(
                "akun",
                "suspend",
                "freeze",
                "dibekukan",
                "diblokir",
                "permanen",
                "fair",
                "fairness",
                "adil",
                "sepihak",
            ),
        ),
        IssueRule(
            label="Privasi dan penyalahgunaan data",
            keywords=(
                "data",
                "privasi",
                "bocor",
                "sebar",
                "akses",
                "kontak",
                "nomor",
                "hack",
                "keamanan",
                "disalahgunakan",
            ),
        ),
        IssueRule(
            label="Komunikasi dan kepastian proses",
            keywords=(
                "respon",
                "balas",
                "komunikasi",
                "informasi",
                "kepastian",
                "pemberitahuan",
                "notifikasi",
                "konfirmasi",
                "janji",
                "kabar",
            ),
        ),
    ),
    "service": (
        IssueRule(
            label="Bug, error, login, dan OTP",
            keywords=(
                "error",
                "bug",
                "crash",
                "login",
                "otp",
                "verifikasi",
                "loading",
                "gagal masuk",
                "gagal login",
                "sistem sibuk",
                "tidak bisa masuk",
                "force close",
            ),
        ),
        IssueRule(
            label="CS dan respon admin",
            keywords=(
                "cs",
                "customer service",
                "service center",
                "admin",
                "respon",
                "balas",
                "pelayanan",
                "bantuan",
                "komplain",
                "chat",
                "support",
                "solusi",
            ),
        ),
        IssueRule(
            label="Proses dan pencairan",
            keywords=(
                "proses",
                "diproses",
                "pencairan",
                "cair",
                "pending",
                "lama",
                "review",
                "menunggu",
                "tidak cair",
                "verifikasi lama",
                "approval",
            ),
        ),
        IssueRule(
            label="Kemudahan penggunaan dan UX",
            keywords=(
                "ribet",
                "sulit",
                "susah",
                "fitur",
                "pakai",
                "gunakan",
                "mudah",
                "membingungkan",
                "navigasi",
                "user friendly",
                "ui",
                "ux",
                "antarmuka",
            ),
        ),
        IssueRule(
            label="Performa dan stabilitas aplikasi",
            keywords=(
                "lambat",
                "lemot",
                "hang",
                "ngelag",
                "lag",
                "stuck",
                "blank",
                "server",
                "maintenance",
                "down",
                "error server",
                "lamban",
            ),
        ),
        IssueRule(
            label="Update dan gangguan aplikasi",
            keywords=(
                "update",
                "upgrade",
                "maintenance",
                "gangguan",
                "downtime",
                "versi",
                "refresh",
                "restart",
            ),
        ),
    ),
}


# Broader keyword lists for presence detection. These are intentionally a bit
# wider than the issue taxonomy so we can cheaply detect whether a review even
# touches an aspect before doing more detailed analysis.
ASPECT_PRESENCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "risk": (
        "limit",
        "pinjaman",
        "utang",
        "hutang",
        "bunga",
        "denda",
        "cicilan",
        "tenor",
        "tagihan",
        "tagih",
        "penagihan",
        "dc",
        "debt collector",
        "cair",
        "pencairan",
        "pelunasan",
        "slik",
        "skor kredit",
        "data",
        "privasi",
        "keamanan",
    ),
    "trust": (
        "percaya",
        "kepercayaan",
        "terpercaya",
        "aman",
        "amanah",
        "legal",
        "legalitas",
        "ojk",
        "resmi",
        "transparan",
        "transparansi",
        "jelas",
        "kejelasan",
        "scam",
        "fraud",
        "tipu",
        "penipu",
        "bohong",
        "privasi",
        "data",
        "keamanan",
    ),
    "service": (
        "cs",
        "customer service",
        "service center",
        "admin",
        "respon",
        "balas",
        "bantuan",
        "pelayanan",
        "login",
        "otp",
        "verifikasi",
        "error",
        "bug",
        "lemot",
        "lambat",
        "fitur",
        "ui",
        "ux",
        "aplikasi",
        "app",
        "apk",
        "loading",
        "crash",
    ),
}


def canonical_aspect(aspect: str) -> str:
    """Return a normalized aspect name and validate that it is supported."""

    normalized = str(aspect).strip().lower()
    if normalized not in ASPECT_ORDER:
        raise ValueError(f"Unknown aspect: {aspect!r}. Expected one of: {', '.join(ASPECT_ORDER)}")
    return normalized


def normalize_text(text: object) -> str:
    """Normalize free text for keyword matching.

    The normalizer is intentionally lightweight: it lowercases the input,
    replaces punctuation with spaces, and collapses repeated whitespace. That
    keeps phrase matching stable without needing a heavier NLP dependency.
    """

    if text is None:
        return ""
    value = str(text).lower()
    value = re.sub(r"[^0-9a-z]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = f" {normalize_text(text)} "
    normalized_keyword = f" {normalize_text(keyword)} "
    return bool(normalized_keyword.strip()) and normalized_keyword in normalized_text


def _matching_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    return [keyword for keyword in keywords if _contains_keyword(text, keyword)]


def issue_rules(aspect: str | None = None) -> tuple[IssueRule, ...] | dict[str, tuple[IssueRule, ...]]:
    """Return the issue taxonomy for one aspect or for all supported aspects."""

    if aspect is None:
        return ISSUE_TAXONOMY
    return ISSUE_TAXONOMY[canonical_aspect(aspect)]


def issue_keywords(aspect: str, label: str | None = None) -> tuple[str, ...]:
    """Return the keyword set for an aspect, or for a specific issue label."""

    rules = ISSUE_TAXONOMY[canonical_aspect(aspect)]
    if label is None:
        merged: list[str] = []
        for rule in rules:
            merged.extend(rule.keywords)
        return tuple(dict.fromkeys(merged))

    for rule in rules:
        if rule.label == label:
            return rule.keywords
    raise KeyError(f"Unknown issue label {label!r} for aspect {aspect!r}")


def assign_issue_label(text: object, aspect: str, default: str = GENERAL_ISSUE_LABEL) -> tuple[str, list[str]]:
    """Assign the strongest issue label and return the matched keywords.

    The function uses a simple "most keyword hits wins" strategy so the output
    stays explainable and easy to audit.
    """

    normalized_aspect = canonical_aspect(aspect)
    best_label = default
    best_hits: list[str] = []
    best_score = 0

    for rule in ISSUE_TAXONOMY[normalized_aspect]:
        hits = _matching_keywords(text if text is not None else "", rule.keywords)
        score = len(hits)
        if score > best_score:
            best_label = rule.label
            best_hits = hits
            best_score = score

    return best_label, best_hits


def aspect_presence(text: object, aspect: str) -> bool:
    """Return ``True`` when a review appears to mention the given aspect."""

    return aspect_presence_hits(text, aspect) > 0


def aspect_presence_hits(text: object, aspect: str) -> int:
    """Return the number of presence keywords matched for an aspect."""

    normalized_aspect = canonical_aspect(aspect)
    return len(_matching_keywords(text if text is not None else "", ASPECT_PRESENCE_KEYWORDS[normalized_aspect]))


def aspect_presence_keywords(text: object, aspect: str) -> tuple[str, ...]:
    """Return the presence keywords matched for a given aspect."""

    normalized_aspect = canonical_aspect(aspect)
    return tuple(_matching_keywords(text if text is not None else "", ASPECT_PRESENCE_KEYWORDS[normalized_aspect]))


def aspect_presence_map(text: object) -> dict[str, bool]:
    """Return a ``{aspect: present}`` map for all supported aspects."""

    return {aspect: aspect_presence(text, aspect) for aspect in ASPECT_ORDER}


def aspect_presence_details(text: object) -> dict[str, dict[str, object]]:
    """Return presence booleans and matched keywords for all supported aspects."""

    return {
        aspect: {
            "present": aspect_presence(text, aspect),
            "hits": aspect_presence_keywords(text, aspect),
        }
        for aspect in ASPECT_ORDER
    }


def aspect_display_name(aspect: str) -> str:
    """Return the dashboard display name for an aspect."""

    normalized = canonical_aspect(aspect)
    return ASPECT_DISPLAY_NAMES[normalized]


__all__ = [
    "ASPECT_DISPLAY_NAMES",
    "ASPECT_ORDER",
    "ASPECT_PRESENCE_KEYWORDS",
    "GENERAL_ISSUE_LABEL",
    "ISSUE_TAXONOMY",
    "IssueRule",
    "aspect_display_name",
    "aspect_presence",
    "aspect_presence_details",
    "aspect_presence_hits",
    "aspect_presence_keywords",
    "aspect_presence_map",
    "assign_issue_label",
    "canonical_aspect",
    "issue_keywords",
    "issue_rules",
    "normalize_text",
]
