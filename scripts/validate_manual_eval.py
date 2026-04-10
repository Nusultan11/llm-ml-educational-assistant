import argparse
import json
import sys
from pathlib import Path

from llm_ml_assistant.utils.eval_validation import load_eval_items, validate_eval_items
from llm_ml_assistant.utils.split_integrity import (
    load_jsonl_rows,
    validate_split_integrity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate independent manual eval JSON file.")
    parser.add_argument("--eval", default="reports/eval/manual_eval_v1.json")
    parser.add_argument("--min-query-chars", type=int, default=8)
    parser.add_argument("--min-expected-chars", type=int, default=8)
    parser.add_argument("--rag", default="")
    parser.add_argument("--sft", default="")
    parser.add_argument("--auto-eval", default="")
    parser.add_argument("--max-leakage-examples", type=int, default=10)

    args = parser.parse_args()

    eval_path = Path(args.eval)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    items = load_eval_items(eval_path)
    summary, errors = validate_eval_items(
        items,
        min_query_chars=args.min_query_chars,
        min_expected_chars=args.min_expected_chars,
    )

    if args.rag or args.sft or args.auto_eval:
        rag_rows = load_jsonl_rows(Path(args.rag)) if args.rag else []
        sft_rows = load_jsonl_rows(Path(args.sft)) if args.sft else []
        auto_eval_items = load_eval_items(Path(args.auto_eval)) if args.auto_eval else []
        split_summary, split_errors = validate_split_integrity(
            rag_rows=rag_rows,
            sft_rows=sft_rows,
            auto_eval_items=auto_eval_items,
            manual_eval_items=items,
            max_examples=args.max_leakage_examples,
        )
        summary["split_integrity"] = split_summary
        errors.extend(split_errors)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        print("\nValidation errors:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("\nEval file is valid.")


if __name__ == "__main__":
    main()
