#!/usr/bin/env python3
"""Verify whether a Vast.ai workspace has received a patch release cleanly.

Typical usage from a release folder:

    python scripts/check_release_sync.py \
      --repo-root /workspace/skripsi_clean \
      --manifest /workspace/releases/skripsi_patch_2026_04_07/release_manifest.json \
      --check-runtime

The checker compares the target files against the expected SHA256 hashes from
the manifest and optionally probes the repo virtualenv for key runtime imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "files" not in payload or not isinstance(payload["files"], list):
        raise ValueError("Manifest JSON must contain a 'files' list.")
    return payload


def detect_python_bin(repo_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    candidates = [
        repo_root / ".venv" / "bin" / "python",
        repo_root / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def probe_runtime(python_bin: Path) -> dict:
    code = r"""
import importlib
import json

payload = {"imports": {}, "torch": None}
for name in ["torch", "peft", "transformers", "bitsandbytes"]:
    try:
        module = importlib.import_module(name)
        payload["imports"][name] = {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
        }
    except Exception as exc:
        payload["imports"][name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

try:
    import torch

    payload["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    payload["torch"] = {
        "error": f"{type(exc).__name__}: {exc}",
    }

print(json.dumps(payload))
"""
    completed = subprocess.run(
        [str(python_bin), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr.strip() or completed.stdout.strip() or "runtime probe failed",
        }
    try:
        parsed = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid runtime probe output: {exc}"}
    parsed["ok"] = True
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a patch release is fully synced into a repo.")
    parser.add_argument("--repo-root", default=".", help="Target repo root to verify.")
    parser.add_argument("--manifest", required=True, help="Path to release_manifest.json.")
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Also probe the repo virtualenv for torch/peft/transformers/bitsandbytes.",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Optional Python interpreter to use for runtime probing. Defaults to repo .venv if found.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()

    manifest = load_manifest(manifest_path)
    release_name = str(manifest.get("release_name", manifest_path.stem))

    print(f"[CHECK] Release: {release_name}")
    print(f"[CHECK] Repo root: {repo_root}")
    print(f"[CHECK] Manifest: {manifest_path}")
    print()

    missing = 0
    mismatched = 0
    ok = 0

    for file_info in manifest["files"]:
        rel_path = Path(file_info["path"])
        expected_hash = str(file_info["sha256"]).lower()
        target_path = repo_root / rel_path

        if not target_path.exists():
            print(f"[MISSING] {rel_path}")
            missing += 1
            continue

        actual_hash = sha256_file(target_path).lower()
        if actual_hash != expected_hash:
            print(f"[MISMATCH] {rel_path}")
            print(f"  expected={expected_hash}")
            print(f"  actual  ={actual_hash}")
            mismatched += 1
            continue

        print(f"[OK] {rel_path}")
        ok += 1

    print()
    print(f"[SUMMARY] ok={ok} missing={missing} mismatched={mismatched}")

    runtime_failed = False
    if args.check_runtime:
        python_bin = detect_python_bin(repo_root, args.python_bin)
        print()
        if python_bin is None or not python_bin.exists():
            print("[RUNTIME] SKIP - Python binary not found. Pass --python-bin if needed.")
        else:
            print(f"[RUNTIME] Using {python_bin}")
            runtime = probe_runtime(python_bin)
            if not runtime.get("ok"):
                print(f"[RUNTIME] FAIL - {runtime.get('error', 'unknown error')}")
                runtime_failed = True
            else:
                imports = runtime.get("imports", {})
                for name in ["torch", "peft", "transformers", "bitsandbytes"]:
                    info = imports.get(name, {})
                    if info.get("ok"):
                        print(f"[RUNTIME] OK   {name:<12} version={info.get('version', 'unknown')}")
                    else:
                        print(f"[RUNTIME] FAIL {name:<12} {info.get('error', 'import failed')}")
                        runtime_failed = True
                torch_info = runtime.get("torch") or {}
                if "error" in torch_info:
                    print(f"[RUNTIME] FAIL torch_probe   {torch_info['error']}")
                    runtime_failed = True
                else:
                    print(
                        "[RUNTIME] INFO torch_probe   "
                        f"cuda_available={torch_info.get('cuda_available')} "
                        f"cuda_version={torch_info.get('cuda_version')} "
                        f"device={torch_info.get('device_name') or '-'}"
                    )

    return 1 if (missing or mismatched or runtime_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
