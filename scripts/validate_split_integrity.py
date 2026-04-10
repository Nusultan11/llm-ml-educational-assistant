import argparse
import json
import sys
from pathlib import Path

from llm_ml_assistant.utils.eval_validation import load_eval_items
from llm_ml_assistant.utils.split_integrity import (
    load_jsonl_rows,
    validate_split_integrity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate split integrity and detect train/eval leakage.")
    parser.add_argument("--rag", default="")
    parser.add_argument("--sft", default="")
    parser.add_argument("--auto-eval", default="")
    parser.add_argument("--manual-eval", default="")
    parser.add_argument("--max-examples", type=int, default=10)

    args = parser.parse_args()

    rag_rows = load_jsonl_rows(Path(args.rag)) if args.rag else []
    sft_rows = load_jsonl_rows(Path(args.sft)) if args.sft else []
    auto_eval_items = load_eval_items(Path(args.auto_eval)) if args.auto_eval else []
    manual_eval_items = load_eval_items(Path(args.manual_eval)) if args.manual_eval else []

    summary, errors = validate_split_integrity(
        rag_rows=rag_rows,
        sft_rows=sft_rows,
        auto_eval_items=auto_eval_items,
        manual_eval_items=manual_eval_items,
        max_examples=args.max_examples,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        print("\nSplit integrity errors:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("\nSplit integrity check passed.")


if __name__ == "__main__":
    main()
