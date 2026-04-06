from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ADALORA_INIT_R_CANDIDATES,
    ADALORA_TARGET_R,
    FULL_FINETUNE_LR_CANDIDATES,
    PEFT_DEFAULT_LR,
    PEFT_DROPOUT_CANDIDATES,
    PEFT_LR_CANDIDATES,
    PEFT_R_CANDIDATES,
    TRAIN_BATCH_SIZE,
    TRAIN_MAX_EPOCHS,
)


def main() -> None:
    payload = {
        "train_max_epochs": TRAIN_MAX_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "full_finetune_lr_candidates": FULL_FINETUNE_LR_CANDIDATES,
        "peft_default_lr": PEFT_DEFAULT_LR,
        "peft_lr_candidates": PEFT_LR_CANDIDATES,
        "peft_r_candidates": PEFT_R_CANDIDATES,
        "peft_dropout_candidates": PEFT_DROPOUT_CANDIDATES,
        "adalora_init_r_candidates": ADALORA_INIT_R_CANDIDATES,
        "adalora_target_r_default": ADALORA_TARGET_R,
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
