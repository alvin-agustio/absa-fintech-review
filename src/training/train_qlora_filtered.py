import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.peft_family_utils import build_variant_parser, run_peft_training


def main():
    parser = build_variant_parser("qlora", filtered=True)
    args = parser.parse_args()
    run_peft_training(args, family_name="qlora", filtered=True)


if __name__ == "__main__":
    main()
