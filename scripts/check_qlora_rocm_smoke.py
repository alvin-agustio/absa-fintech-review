from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from config import BASE_MODEL_NAME, MAX_LENGTH, QLORA_COMPUTE_DTYPE, QLORA_DOUBLE_QUANT, QLORA_QUANT_TYPE  # noqa: E402


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        output = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
        return completed.returncode, output.strip()
    except Exception as exc:
        return 1, str(exc)


def safe_import(module_name: str):
    try:
        module = importlib.import_module(module_name)
        return True, getattr(module, "__version__", "unknown")
    except Exception as exc:
        return False, str(exc)


def detect_backend() -> dict:
    cuda_available = bool(torch.cuda.is_available())
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    backend = "cpu"
    if hip_version:
        backend = "rocm"
    elif cuda_version:
        backend = "cuda"

    device_name = None
    total_memory_gb = None
    if cuda_available:
        try:
            props = torch.cuda.get_device_properties(0)
            device_name = props.name
            total_memory_gb = round(props.total_memory / (1024 ** 3), 2)
        except Exception:
            device_name = "unavailable"

    return {
        "cuda_available": cuda_available,
        "backend": backend,
        "hip_version": hip_version,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "device_memory_gb": total_memory_gb,
    }


def perform_checks(model_name: str, batch_size: int, max_length: int) -> dict:
    checks: list[CheckResult] = []

    checks.append(CheckResult("python", "PASS", platform.python_version()))

    torch_ok = True
    try:
        torch_version = torch.__version__
    except Exception as exc:
        torch_ok = False
        torch_version = str(exc)
    checks.append(CheckResult("torch_import", "PASS" if torch_ok else "FAIL", torch_version))

    backend = detect_backend()
    if backend["cuda_available"]:
        checks.append(
            CheckResult(
                "gpu_runtime",
                "PASS",
                f"backend={backend['backend']} device={backend['device_name']} memory_gb={backend['device_memory_gb']}",
            )
        )
    else:
        checks.append(CheckResult("gpu_runtime", "FAIL", "torch.cuda.is_available() == False"))

    peft_ok, peft_detail = safe_import("peft")
    checks.append(CheckResult("peft_import", "PASS" if peft_ok else "FAIL", peft_detail))

    transformers_ok, transformers_detail = safe_import("transformers")
    checks.append(CheckResult("transformers_import", "PASS" if transformers_ok else "FAIL", transformers_detail))

    bnb_ok, bnb_detail = safe_import("bitsandbytes")
    if bnb_ok:
        checks.append(CheckResult("bitsandbytes_import", "PASS", bnb_detail))
    else:
        checks.append(
            CheckResult(
                "bitsandbytes_import",
                "FAIL",
                f"{bnb_detail}. QLoRA 4-bit path usually depends on bitsandbytes.",
            )
        )

    rocm_info_rc, rocm_info_out = run_cmd(["rocminfo"])
    checks.append(
        CheckResult(
            "rocminfo",
            "PASS" if rocm_info_rc == 0 else "WARN",
            rocm_info_out[:500] if rocm_info_out else "rocminfo not available",
        )
    )

    amd_smi_rc, amd_smi_out = run_cmd(["amd-smi", "static", "--gpu", "0"])
    checks.append(
        CheckResult(
            "amd_smi",
            "PASS" if amd_smi_rc == 0 else "WARN",
            amd_smi_out[:500] if amd_smi_out else "amd-smi not available",
        )
    )

    qlora_load_status = "FAIL"
    qlora_load_detail = "Preconditions not met."
    qlora_forward_status = "FAIL"
    qlora_forward_detail = "Model load did not complete."

    if backend["cuda_available"] and transformers_ok and peft_ok and bnb_ok:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=QLORA_QUANT_TYPE,
                bnb_4bit_use_double_quant=QLORA_DOUBLE_QUANT,
                bnb_4bit_compute_dtype=getattr(torch, QLORA_COMPUTE_DTYPE, torch.float16),
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                quantization_config=quant_config,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                bias="none",
            )
            model = get_peft_model(model, config)
            qlora_load_status = "PASS"
            qlora_load_detail = f"Loaded 4-bit model and attached LoRA adapters for {model_name}"

            encoded = tokenizer(
                ["[ASPECT=risk] aplikasi cukup baik", "[ASPECT=service] proses cepat"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            device = torch.device("cuda")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            model.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(**encoded)
            logits_shape = tuple(outputs.logits.shape)
            qlora_forward_status = "PASS"
            qlora_forward_detail = f"Forward pass ok with logits shape={logits_shape}, batch_size={batch_size}"
        except Exception as exc:
            tb = traceback.format_exc(limit=1)
            qlora_load_detail = f"{type(exc).__name__}: {exc}"
            qlora_forward_detail = tb.strip()

    checks.append(CheckResult("qlora_4bit_load", qlora_load_status, qlora_load_detail))
    checks.append(CheckResult("qlora_forward_pass", qlora_forward_status, qlora_forward_detail))

    overall_status = "PASS"
    if any(check.status == "FAIL" for check in checks):
        overall_status = "FAIL"
    elif any(check.status == "WARN" for check in checks):
        overall_status = "WARN"

    recommendation = {
        "PASS": "QLoRA smoke test looks healthy. You can continue to a short trial run before full sweep.",
        "WARN": "QLoRA smoke test is partially healthy, but there are warnings. Fix them before the full run if they affect quantization or ROCm visibility.",
        "FAIL": "Do not start the full QLoRA run yet. Fix the failed checks first.",
    }[overall_status]

    return {
        "generated_at_utc": iso_utc_now(),
        "model_name": model_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "backend": backend,
        "checks": [asdict(check) for check in checks],
        "overall_status": overall_status,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test QLoRA readiness on MI300X/ROCm before full training.")
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--output_json", default=str(ROOT / "data" / "processed" / "evaluation" / "qlora_rocm_smoke_test.json"))
    args = parser.parse_args()

    result = perform_checks(args.model_name, args.batch_size, args.max_length)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["overall_status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
