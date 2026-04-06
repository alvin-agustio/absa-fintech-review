from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.aspect_taxonomy import ASPECT_DISPLAY_NAMES, ASPECT_ORDER  # noqa: E402
from src.dashboard.summary_rules import build_summary_payload  # noqa: E402


CACHE_DIR = ROOT / "data" / "processed" / "dashboard" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "audits" / "summary_rules"


def _latest_cache_pair(cache_dir: Path) -> tuple[Path, Path] | None:
    candidates: list[tuple[float, Path, Path]] = []
    for reviews_path in cache_dir.glob("*_reviews.csv"):
        preds_path = reviews_path.with_name(reviews_path.name.replace("_reviews.csv", "_predictions.csv"))
        if preds_path.exists():
            candidates.append((reviews_path.stat().st_mtime, reviews_path, preds_path))
    if not candidates:
        return None
    _, reviews_path, preds_path = max(candidates, key=lambda item: item[0])
    return reviews_path, preds_path


def _merge_cache_pair(reviews_path: Path, preds_path: Path) -> pd.DataFrame:
    reviews = pd.read_csv(reviews_path)
    preds = pd.read_csv(preds_path)
    long_df = reviews.merge(preds, on=["review_id_ext", "source_job_id"], how="inner")
    if "review_date" in long_df.columns:
        long_df["review_date"] = pd.to_datetime(long_df["review_date"], errors="coerce")
    return long_df


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if is_dataclass(payload):
        return asdict(payload)
    raise TypeError(f"Unsupported payload type: {type(payload)!r}")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    if value is pd.NA:
        return None
    return value


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()


def _guardrails_from_payload(payload: dict[str, Any], long_df: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    apps = [str(app.get("app_name", "")).strip() for app in payload.get("apps", []) if isinstance(app, dict)]
    app_metrics = [app.get("metrics", {}) for app in payload.get("apps", []) if isinstance(app, dict)]
    app_bodies = [str(app.get("body", app.get("text", ""))) for app in payload.get("apps", []) if isinstance(app, dict)]

    has_kredivo = "Kredivo" in apps
    has_akulaku = "Akulaku" in apps
    checks.append(
        {
            "name": "both_apps_present",
            "status": "pass" if has_kredivo and has_akulaku else "warn",
            "detail": "Kredivo dan Akulaku sama-sama muncul di summary." if has_kredivo and has_akulaku else "Summary hanya menampilkan satu aplikasi atau urutan app belum lengkap.",
        }
    )

    checks.append(
        {
            "name": "app_order_is_kredivo_first",
            "status": "pass" if apps[:2] == ["Kredivo", "Akulaku"] or len(apps) < 2 else "warn",
            "detail": "Urutan app sudah Kredivo lalu Akulaku." if apps[:2] == ["Kredivo", "Akulaku"] else "Urutan app perlu dicek lagi agar konsisten dengan default dashboard.",
        }
    )

    app_name_matches = True
    for app in payload.get("apps", []):
        if not isinstance(app, dict):
            continue
        app_name = str(app.get("app_name", "")).strip()
        body = str(app.get("body", app.get("text", "")))
        if app_name and app_name not in body:
            app_name_matches = False
            break
    checks.append(
        {
            "name": "app_cards_name_anchored",
            "status": "pass" if app_name_matches else "warn",
            "detail": "Setiap kartu app menyebut nama aplikasinya sendiri." if app_name_matches else "Ada kartu app yang belum menyebut nama aplikasinya secara eksplisit.",
        }
    )

    overall = payload.get("overall", {}) if isinstance(payload.get("overall"), dict) else {}
    signal = payload.get("signal", {}) if isinstance(payload.get("signal"), dict) else {}
    meaning = payload.get("meaning", {}) if isinstance(payload.get("meaning"), dict) else {}

    best_aspect = str(overall.get("metrics", {}).get("best_aspect", "")).strip()
    worst_aspect = str(overall.get("metrics", {}).get("worst_aspect", "")).strip()
    overall_body = str(overall.get("body", overall.get("text", "")))
    signal_body = str(signal.get("body", signal.get("text", "")))
    meaning_body = str(meaning.get("body", meaning.get("text", "")))

    best_label = ASPECT_DISPLAY_NAMES.get(best_aspect, best_aspect.title()) if best_aspect else ""
    worst_label = ASPECT_DISPLAY_NAMES.get(worst_aspect, worst_aspect.title()) if worst_aspect else ""

    checks.append(
        {
            "name": "overall_mentions_anchor_aspects",
            "status": "pass" if best_label in overall_body and worst_label in overall_body else "warn",
            "detail": f"Overall summary mengikat aspek kuat ({best_label}) dan aspek lemah ({worst_label}).",
        }
    )

    checks.append(
        {
            "name": "signal_mentions_worst_aspect",
            "status": "pass" if worst_label in signal_body else "warn",
            "detail": f"Blok sinyal menyorot aspek {worst_label} sebagai prioritas.",
        }
    )

    checks.append(
        {
            "name": "meaning_mentions_app_contrast",
            "status": "pass" if ("Kredivo" in meaning_body and "Akulaku" in meaning_body) or len(apps) < 2 else "warn",
            "detail": "Makna akhir sudah membedakan Kredivo dan Akulaku." if ("Kredivo" in meaning_body and "Akulaku" in meaning_body) or len(apps) < 2 else "Makna akhir belum cukup eksplisit membedakan dua app.",
        }
    )

    if len(app_bodies) >= 2:
        similarity = _similarity(app_bodies[0], app_bodies[1])
        negative_delta = 0.0
        positive_delta = 0.0
        if len(app_metrics) >= 2:
            left_metrics = app_metrics[0] or {}
            right_metrics = app_metrics[1] or {}
            negative_delta = abs(float(left_metrics.get("negative_share", 0.0)) - float(right_metrics.get("negative_share", 0.0)))
            positive_delta = abs(float(left_metrics.get("positive_share", 0.0)) - float(right_metrics.get("positive_share", 0.0)))
        contrast_pass = negative_delta >= 5.0 or positive_delta >= 5.0
        checks.append(
            {
                "name": "app_contrast_visible",
                "status": "pass" if contrast_pass else "warn",
                "detail": f"Kontras app terlihat lewat gap sentimen ({positive_delta:.1f} pts positif, {negative_delta:.1f} pts negatif), similarity teks {similarity:.2f}.",
            }
        )
    else:
        checks.append(
            {
                "name": "app_contrast_visible",
                "status": "warn",
                "detail": "Hanya ada satu kartu app, jadi kontras Kredivo vs Akulaku tidak bisa diaudit.",
            }
        )

    checks.append(
        {
            "name": "issue_column_available",
            "status": "info" if not bool(payload.get("coverage", {}).get("has_issue_column")) else "pass",
            "detail": "Kolom issue tersedia dan summary bisa lebih spesifik." if bool(payload.get("coverage", {}).get("has_issue_column")) else "Kolom issue tidak tersedia; summary memakai label aspek/hint sebagai fallback.",
        }
    )

    if payload.get("status") == "insufficient_data":
        checks.append(
            {
                "name": "data_sufficiency",
                "status": "warn",
                "detail": "Payload masih terlalu kecil, jadi summary sebaiknya dibaca sebagai indikasi awal.",
            }
        )
    else:
        checks.append(
            {
                "name": "data_sufficiency",
                "status": "pass",
                "detail": "Ukuran data cukup untuk summary ringkas dan perbandingan app.",
            }
        )

    return checks


def build_audit_report(long_df: pd.DataFrame, *, source_label: str = "manual") -> dict[str, Any]:
    payload = _normalize_payload(build_summary_payload(long_df))
    guardrails = _guardrails_from_payload(payload, long_df)

    app_rows = []
    for app in payload.get("apps", []):
        if not isinstance(app, dict):
            continue
        metrics = app.get("metrics", {}) if isinstance(app.get("metrics"), dict) else {}
        app_rows.append(
            {
                "app_name": app.get("app_name"),
                "tone": app.get("tone"),
                "dominant_sentiment": metrics.get("dominant_sentiment"),
                "positive_share": metrics.get("positive_share"),
                "neutral_share": metrics.get("neutral_share"),
                "negative_share": metrics.get("negative_share"),
                "best_aspect": metrics.get("best_aspect"),
                "worst_aspect": metrics.get("worst_aspect"),
                "body": app.get("body", app.get("text", "")),
            }
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_label": source_label,
        "coverage": payload.get("coverage", {}),
        "status": payload.get("status"),
        "warnings": payload.get("warnings", []),
        "guardrails": guardrails,
        "summary": {
            "overall": payload.get("overall", {}),
            "signal": payload.get("signal", {}),
            "meaning": payload.get("meaning", {}),
            "apps": app_rows,
        },
    }
    return report


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        cells = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report.get("coverage", {})
    summary = report.get("summary", {})
    guardrails = report.get("guardrails", [])
    apps = summary.get("apps", [])
    overall = summary.get("overall", {})
    signal = summary.get("signal", {})
    meaning = summary.get("meaning", {})

    lines = [
        "# Summary Rules Audit",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Source: `{report.get('source_label', '')}`",
        f"- Status: `{report.get('status', '')}`",
        f"- Unique reviews: `{coverage.get('unique_reviews', 0)}`",
        f"- Apps: `{', '.join(coverage.get('apps', [])) or 'n/a'}`",
        f"- Aspects: `{', '.join(coverage.get('aspects', [])) or 'n/a'}`",
        f"- Quality: `{coverage.get('quality', 'n/a')}`",
        "",
        "## Warnings",
    ]
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- No warnings.")

    lines.extend(
        [
            "",
            "## Guardrails",
        ]
    )
    for guardrail in guardrails:
        lines.append(f"- [{guardrail.get('status', 'info').upper()}] {guardrail.get('name', '')}: {guardrail.get('detail', '')}")

    lines.extend(
        [
            "",
            "## Summary Preview",
            "",
            f"**Overall**: {overall.get('body', overall.get('text', ''))}",
            "",
            f"**Signal**: {signal.get('body', signal.get('text', ''))}",
            "",
            f"**Meaning**: {meaning.get('body', meaning.get('text', ''))}",
            "",
            "### Per-App Snapshot",
            "",
        ]
    )
    lines.append(
        _markdown_table(
            apps,
            ["app_name", "tone", "dominant_sentiment", "positive_share", "neutral_share", "negative_share", "best_aspect", "worst_aspect"],
        )
    )
    lines.append("")
    lines.append("## Per-App Text")
    for app in apps:
        lines.append(f"### {app.get('app_name', 'App')}")
        lines.append(str(app.get("body", "")))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    json_path = output_dir / f"summary_rules_audit_{stamp}.json"
    md_path = output_dir / f"summary_rules_audit_{stamp}.md"
    latest_json = output_dir / "summary_rules_audit_latest.json"
    latest_md = output_dir / "summary_rules_audit_latest.md"
    json_text = json.dumps(_jsonable(report), indent=2, ensure_ascii=False)
    md_text = render_markdown(report)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    return md_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the rule-based summary payload on dashboard cache data.")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="Dashboard cache directory to inspect.")
    parser.add_argument("--reviews", type=Path, default=None, help="Explicit reviews CSV path.")
    parser.add_argument("--predictions", type=Path, default=None, help="Explicit predictions CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write the audit report.")
    args = parser.parse_args()

    if args.reviews and args.predictions:
        reviews_path = args.reviews
        preds_path = args.predictions
        source_label = f"{reviews_path.name} + {preds_path.name}"
    else:
        pair = _latest_cache_pair(args.cache_dir)
        if pair is None:
            print("No dashboard cache pair found.", file=sys.stderr)
            return 1
        reviews_path, preds_path = pair
        source_label = f"latest cache pair: {reviews_path.name} + {preds_path.name}"

    long_df = _merge_cache_pair(reviews_path, preds_path)
    report = build_audit_report(long_df, source_label=source_label)
    md_path, json_path = _write_report(report, args.output_dir)
    print(f"Wrote audit report to {md_path}")
    print(f"Wrote audit JSON to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
