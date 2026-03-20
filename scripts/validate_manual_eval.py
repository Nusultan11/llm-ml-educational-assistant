import argparse
import json
import sys
from pathlib import Path

from llm_ml_assistant.utils.eval_validation import load_eval_items, validate_eval_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate independent manual eval JSON file.")
    parser.add_argument("--eval", default="reports/eval/manual_eval_v1.json")
    parser.add_argument("--min-query-chars", type=int, default=8)
    parser.add_argument("--min-expected-chars", type=int, default=8)

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

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if errors:
        print("\nValidation errors:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("\nEval file is valid.")


if __name__ == "__main__":
    main()
