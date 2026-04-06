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
    priority: int = 0


# Stronger, broader taxonomy than the inline dashboard rules.
# The labels are meant to be readable in cards and diagnostics.
ISSUE_TAXONOMY: dict[str, tuple[IssueRule, ...]] = {
    "risk": (
        IssueRule(
            label="Limit, approval, dan pencairan",
            priority=5,
            keywords=(
                "limit",
                "limit naik",
                "limit turun",
                "naik limit",
                "turun limit",
                "limit tidak tersedia",
                "limit kosong",
                "ditolak",
                "pengajuan ditolak",
                "ditolak terus",
                "tidak lolos",
                "tidak pernah lolos",
                "penolakan",
                "tolak",
                "approval",
                "persetujuan",
                "tidak disetujui",
                "gagal approve",
                "cair",
                "pencairan",
                "dana cair",
                "belum cair",
                "gagal cair",
                "pencairan gagal",
                "pinjaman",
                "score kredit",
                "skor kredit",
                "slik",
                "pernah telat",
                "pinjam lagi",
                "bisa pinjam",
                "limit paylater",
                "tidak bisa pinjam",
                "tidak bisa pakai limit",
            ),
        ),
        IssueRule(
            label="Bunga, biaya, dan denda",
            priority=4,
            keywords=(
                "bunga",
                "bunga tinggi",
                "biaya",
                "biaya admin",
                "biaya layanan",
                "admin",
                "utang",
                "hutang",
                "tidak merasa hutang",
                "riwayat hutang",
                "bi check",
                "kol 5",
                "denda",
                "denda tinggi",
                "tagihan",
                "cicilan",
                "tenor",
                "pelunasan",
                "mahal",
                "riba",
                "mencekik",
                "potongan",
                "jatuh tempo",
                "telat bayar",
                "terlambat bayar",
                "bayar tepat waktu",
            ),
        ),
        IssueRule(
            label="Penagihan dan debt collector",
            priority=4,
            keywords=(
                "penagihan",
                "tagih",
                "dc",
                "debt collector",
                "debt",
                "telepon terus",
                "hubungi terus",
                "telepon",
                "ancam",
                "ancaman",
                "teror",
                "kasar",
                "sebar data",
                "data sebar",
                "kontak keluarga",
                "hubungi keluarga",
                "wa",
                "whatsapp",
                "sms",
                "surat",
            ),
        ),
        IssueRule(
            label="Blokir setelah pembayaran",
            priority=3,
            keywords=(
                "blokir akun",
                "akun diblokir",
                "akun dibekukan",
                "pelunasan",
                "lunas",
                "bayar cepat",
                "setelah bayar",
                "tidak bisa tutup akun",
                "hapus akun",
                "penutupan akun",
                "blacklist",
            ),
        ),
        IssueRule(
            label="Keamanan data pribadi",
            priority=3,
            keywords=(
                "data pribadi",
                "privasi",
                "kontak darurat",
                "nomor kontak",
                "izin akses",
                "akses kontak",
                "bocor data",
                "data bocor",
                "sebar data",
                "disalahgunakan",
                "penyalahgunaan",
                "hapus data",
            ),
        ),
    ),
    "trust": (
        IssueRule(
            label="Transparansi dan kejelasan",
            priority=4,
            keywords=(
                "tidak jelas",
                "ga jelas",
                "gak jelas",
                "gajelas",
                "jelasnya",
                "tanpa alasan",
                "tanpa sebab",
                "tanpa penjelasan",
                "alasan penolakan",
                "tidak ada alasan",
                "transparan",
                "transparansi",
                "kejelasan",
                "sepihak",
                "status",
                "keterangan",
                "pemberitahuan",
                "notifikasi",
                "konfirmasi",
                "janji",
                "kepastian",
                "kenapa",
                "mengapa",
                "informasi proses",
                "diblok tanpa sebab",
                "ditolak tanpa alasan",
            ),
        ),
        IssueRule(
            label="Legalitas, OJK, dan reputasi",
            priority=2,
            keywords=(
                "legal",
                "legalitas",
                "ojk",
                "resmi",
                "terpercaya",
                "amanah",
                "berizin",
                "terdaftar",
                "reputasi",
                "perusahaan",
            ),
        ),
        IssueRule(
            label="Penipuan dan fraud",
            priority=3,
            keywords=(
                "tipu",
                "menipu",
                "tipuan",
                "penipu",
                "penipuan",
                "pencuri",
                "pencuri uang",
                "scam",
                "hoax",
                "bohong",
                "palsu",
                "fraud",
                "modus",
                "oknum",
                "uang ditarik",
                "uang saya ditarik",
                "ditarik terus",
                "hati hati",
                "jangan download",
                "abal abal",
                "ilegal",
                "riba",
            ),
        ),
        IssueRule(
            label="Peretasan dan keamanan akun",
            priority=4,
            keywords=(
                "retas",
                "diretas",
                "akun di retas",
                "akun diretas",
                "bobol",
                "dibobol",
                "akun dibobol",
                "hack",
                "keamanan sistem",
                "sistem keamanan",
                "bocor",
            ),
        ),
        IssueRule(
            label="Akun, fairness, dan suspend",
            priority=5,
            keywords=(
                "blokir akun",
                "akun diblokir",
                "akun dibekukan",
                "permanen",
                "hapus akun",
                "penutupan akun",
                "tidak bisa login",
                "fair",
                "fairness",
                "adil",
                "sepihak",
                "suspend akun",
                "freeze akun",
            ),
        ),
        IssueRule(
            label="Privasi dan penyalahgunaan data",
            priority=4,
            keywords=(
                "data pribadi",
                "privasi",
                "sebar data",
                "akses kontak",
                "izin akses",
                "nomor kontak",
                "kontak darurat",
                "hack",
                "disalahgunakan",
                "salah gunakan",
                "di salah gunakan",
                "jangan salah gunakan",
                "minta hapus data",
                "hapus data",
                "daftar kontak",
                "bocor data",
                "data bocor",
            ),
        ),
        IssueRule(
            label="Komunikasi dan kepastian proses",
            priority=3,
            keywords=(
                "respon admin",
                "respon lambat",
                "tidak dibalas",
                "balasan",
                "balas pesan",
                "komunikasi",
                "kepastian",
                "pemberitahuan",
                "notifikasi",
                "konfirmasi",
                "janji",
                "kabar",
                "chat cs",
                "customer service",
                "informasi proses",
            ),
        ),
    ),
    "service": (
        IssueRule(
            label="Bug, error, dan stabilitas",
            priority=5,
            keywords=(
                "error",
                "bug",
                "crash",
                "loading",
                "loading lama",
                "loading terus",
                "gagal masuk",
                "gagal login",
                "sistem sibuk",
                "tidak bisa masuk",
                "tidak bisa login",
                "tidak bisa buka",
                "tidak kebuka",
                "force close",
                "hang",
                "stuck",
                "blank",
                "server",
                "maintenance",
                "down",
                "lemot",
                "lambat",
                "gagal",
            ),
        ),
        IssueRule(
            label="Pendaftaran, login, dan verifikasi",
            priority=4,
            keywords=(
                "download",
                "sudah download",
                "daftar",
                "registrasi",
                "isi data",
                "isi data sesuai",
                "login",
                "masuk",
                "verifikasi",
                "otp",
                "kode otp",
                "kode verifikasi",
                "aktivasi",
                "aktivasi akun",
                "verifikasi gagal",
                "gagal daftar",
                "password",
                "username",
                "sandi",
                "akun baru",
                "pengguna baru",
                "email",
                "nomor hp",
            ),
        ),
        IssueRule(
            label="CS dan respon admin",
            priority=4,
            keywords=(
                "cs",
                "customer service",
                "service center",
                "respon admin",
                "respon cs",
                "respon lambat",
                "mohon konfirmasi",
                "minta konfirmasi",
                "konfirmasi ke saya",
                "tidak konfirmasi",
                "konfirmasi penerimaan",
                "balas",
                "balasan",
                "tidak dibalas",
                "balas chat",
                "pelayanan",
                "bantuan",
                "komplain",
                "chat",
                "support",
                "solusi",
            ),
        ),
        IssueRule(
            label="Fitur, pencarian, dan katalog",
            priority=3,
            keywords=(
                "fitur pencarian",
                "pencarian",
                "hasil pencarian",
                "menampilkan barang",
                "cari barang",
                "cari produk",
                "fitur otomatis",
                "konfirmasi penerimaan",
                "penerimaan paket",
                "paket sudah keterima",
                "katalog",
                "katalog produk",
                "produk",
                "barang",
            ),
        ),
        IssueRule(
            label="Proses dan pencairan",
            priority=3,
            keywords=(
                "proses",
                "diproses",
                "pencairan",
                "cair",
                "belum cair",
                "gagal cair",
                "pending",
                "lama",
                "review",
                "menunggu",
                "tidak cair",
                "pengajuan pinjaman",
                "proses pengajuan",
                "limit pinjaman",
                "dapat limit",
                "malah ditolak",
                "verifikasi lama",
                "kesalahan sistem",
                "proses pinjaman",
                "transaksi",
                "pembayaran",
                "sudah dibayar",
                "tidak masuk",
                "masuk rekening",
                "transfer",
            ),
        ),
        IssueRule(
            label="Kemudahan penggunaan dan UX",
            priority=2,
            keywords=(
                "ribet",
                "sulit",
                "susah",
                "susah dipakai",
                "sulit dipakai",
                "membingungkan",
                "navigasi",
                "menu",
                "tampilan",
                "user friendly",
                "antarmuka",
                "mudah digunakan",
                "mudah dipakai",
            ),
        ),
        IssueRule(
            label="Performa dan stabilitas aplikasi",
            priority=2,
            keywords=(
                "lambat",
                "lemot",
                "loading lama",
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
                "berat",
            ),
        ),
        IssueRule(
            label="Update dan gangguan aplikasi",
            priority=1,
            keywords=(
                "update",
                "upgrade",
                "setelah update",
                "versi baru",
                "ganti versi",
                "maintenance",
                "gangguan",
                "downtime",
                "versi",
                "refresh",
                "restart",
                "error setelah update",
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
        "approval",
        "persetujuan",
        "ditolak",
        "pengajuan",
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
        "retas",
        "diretas",
        "bobol",
        "privasi",
        "data",
        "keamanan",
    ),
    "service": (
        "cs",
        "customer service",
        "service center",
        "admin",
        "download",
        "isi data",
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
        "fitur pencarian",
        "pencarian",
        "katalog",
        "produk",
        "barang",
        "ui",
        "ux",
        "aplikasi",
        "app",
        "apk",
        "loading",
        "crash",
        "konfirmasi",
        "penerimaan paket",
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


def _keyword_weight(keyword: str) -> int:
    """Weight a matched keyword by phrase specificity."""

    return max(len(normalize_text(keyword).split()), 1)


def _rule_match_signature(rule: IssueRule, hits: list[str], rule_index: int) -> tuple[int, int, int, int, int, int]:
    """Return a comparison tuple for matching a rule.

    Longer phrases are preferred, then rule priority, then the number of
    matched phrases, and finally the taxonomy order. This keeps ambiguous
    edge cases explainable while still letting more specific rules win.
    """

    score = sum(_keyword_weight(hit) for hit in hits)
    specificity = max((_keyword_weight(hit) for hit in hits), default=0)
    multiword_hits = sum(1 for hit in hits if len(normalize_text(hit).split()) > 1)
    return score, specificity, rule.priority, multiword_hits, len(hits), -rule_index


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
    stays explainable and easy to audit. Longer matched phrases receive a tiny
    boost so more specific issue labels can win over very generic ones, and
    explicit rule priority helps break ties on edge cases.
    """

    normalized_aspect = canonical_aspect(aspect)
    best_label = default
    best_hits: list[str] = []
    best_signature = (-1, -1, -1, -1, -1, -1)

    for rule_index, rule in enumerate(ISSUE_TAXONOMY[normalized_aspect]):
        hits = _matching_keywords(text if text is not None else "", rule.keywords)
        if not hits:
            continue
        candidate = _rule_match_signature(rule, hits, rule_index)
        if candidate > best_signature:
            best_label = rule.label
            best_hits = hits
            best_signature = candidate

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
