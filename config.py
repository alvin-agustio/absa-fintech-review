"""
Global configuration for the ABSA pipeline.
"""

from pathlib import Path

# ── PROJECT PATHS ─────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# ── ACTIVE DATA SOURCES (SOURCE OF TRUTH) ───────────────────────────
# Silver v1 (archived provenance)
DATASET_ABSA_V1_ARCHIVE = DATA_PROCESSED / "archive" / "2026-03-13_focus_v2" / "dataset_absa.csv"

# Current normalized corpora
REVIEWS_CLEAN_V2_PATH = DATA_PROCESSED / "reviews_clean_v2.csv"
DATASET_ABSA_V2_PATH = DATA_PROCESSED / "dataset_absa_v2.csv"

# Official training dataset for experiments (v2 + 50k cohort intersection)
TRAIN_DATASET_PATH = DATA_PROCESSED / "dataset_absa_50k_v2_intersection.csv"
TRAIN_MANIFEST_PATH = DATA_PROCESSED / "manifests" / "stratified_50k_seed42_v2_intersection.csv"

# Gold annotation template currently in use
GOLD_TEMPLATE_PATH = DATA_PROCESSED / "diamond" / "template_anotator_tunggal_balanced_300_aspect_rows.csv"

# ── APP CONFIG ────────────────────────────────────────────────────────
APPS = {
    "Kredivo": "com.finaccel.android",
    "Akulaku": "io.silvrr.installment",
}

LANG = "id"
COUNTRY = "id"

# ── DATE RANGE ────────────────────────────────────────────────────────
DATE_START = "2024-01-01"
DATE_END = "2026-01-31"

# ── LABEL CONFIG ──────────────────────────────────────────────────────
ASPECTS = ["risk", "trust", "service"]

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# ── MODEL CONFIG ──────────────────────────────────────────────────────
BASE_MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128
SEED = 42
TRAIN_MAX_EPOCHS = 15
TRAIN_BATCH_SIZE = 8

FULL_FINETUNE_DEFAULT_LR = 2e-5
FULL_FINETUNE_LR_CANDIDATES = [2e-5, 3e-5, 5e-5]

PEFT_DEFAULT_LR = 2e-4
PEFT_LR_CANDIDATES = [1e-4, 2e-4, 3e-4]
PEFT_R_CANDIDATES = [8, 16]
PEFT_DROPOUT_CANDIDATES = [0.05, 0.1]
ADALORA_INIT_R_CANDIDATES = [12, 16]

# ── LORA CONFIG ───────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]

DORA_R = LORA_R
DORA_ALPHA = LORA_ALPHA
DORA_DROPOUT = LORA_DROPOUT
DORA_TARGET_MODULES = LORA_TARGET_MODULES

ADALORA_INIT_R = 12
ADALORA_TARGET_R = 8
ADALORA_ALPHA = LORA_ALPHA
ADALORA_DROPOUT = LORA_DROPOUT
ADALORA_TARGET_MODULES = LORA_TARGET_MODULES
ADALORA_TINIT = 0
ADALORA_TFINAL = 0
ADALORA_DELTA_T = 1
ADALORA_BETA1 = 0.85
ADALORA_BETA2 = 0.85
ADALORA_ORTH_REG_WEIGHT = 0.5

QLORA_R = LORA_R
QLORA_ALPHA = LORA_ALPHA
QLORA_DROPOUT = LORA_DROPOUT
QLORA_TARGET_MODULES = LORA_TARGET_MODULES
QLORA_QUANT_TYPE = "nf4"
QLORA_DOUBLE_QUANT = True
QLORA_COMPUTE_DTYPE = "float16"

# ── GEMINI CONFIG ─────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BATCH_SIZE = 10  # reviews per API call

# ── KIMI CONFIG ───────────────────────────────────────────────────────
KIMI_MODEL = "moonshot-v1-8k"
KIMI_BATCH_SIZE = 50  # reviews per API call

# ── GROQ CONFIG ───────────────────────────────────────────────────────
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_BATCH_SIZE = 10  # conservative default for Groq TPM limits with full JSON batch labeling

# ── SUMOPOD CONFIG ────────────────────────────────────────────────────
SUMOPOD_MODEL = "openlimit/claude-haiku-4-5"
SUMOPOD_BATCH_SIZE = 50
