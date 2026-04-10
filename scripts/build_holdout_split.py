import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_ml_assistant.utils.data_split import build_holdout_split


def parse_quality_policy(value: str) -> dict[str, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    policy: dict[str, float] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid quality policy segment: {part}")
        key, raw_value = [chunk.strip().lower() for chunk in part.split("=", 1)]
        if key not in {"high", "medium", "low"}:
            raise ValueError(f"Invalid quality bucket in policy: {key}")
        score = float(raw_value)
        if score < 0 or score > 1:
            raise ValueError(f"Policy value must be between 0 and 1: {part}")
        policy[key] = score

    for bucket in ("high", "medium", "low"):
        policy.setdefault(bucket, 1.0)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/dev/eval_holdout split from cleaned SFT data."
    )
    parser.add_argument("--sft-path", default="data/processed_v2_clean/sft_instructions.jsonl")
    parser.add_argument("--out-dir", default="data/processed_v2_clean/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-quality-policy", default="high=1.0,medium=0.8,low=0.35")
    parser.add_argument("--dev-quality-policy", default="high=1.0,medium=0.7,low=0.3")
    parser.add_argument("--eval-quality-policy", default="high=1.0,medium=0.85,low=0.5")

    args = parser.parse_args()

    summary = build_holdout_split(
        sft_path=Path(args.sft_path),
        out_dir=Path(args.out_dir),
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        quality_policies={
            "train": parse_quality_policy(args.train_quality_policy),
            "dev": parse_quality_policy(args.dev_quality_policy),
            "eval_holdout": parse_quality_policy(args.eval_quality_policy),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
