import argparse
import json
import random
import re
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def first_snippet(text: str, min_len: int, max_len: int) -> str:
    t = normalize(text)
    if len(t) <= max_len:
        return t

    cut = t[:max_len]
    for sep in [". ", "? ", "! ", "; "]:
        idx = cut.rfind(sep)
        if idx >= min_len:
            return cut[: idx + 1]

    space_idx = cut.rfind(" ")
    if space_idx >= min_len:
        return cut[:space_idx]
    return cut


def build_eval(
    sft_path: Path,
    out_path: Path,
    max_items: int,
    min_instruction_chars: int,
    min_response_chars: int,
    min_expected_chars: int,
    max_expected_chars: int,
    seed: int,
) -> dict:
    rows = []
    with sft_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(rows)

    eval_items = []
    for row in rows:
        instruction = normalize(row.get("instruction", ""))
        response = normalize(row.get("response", ""))

        if len(instruction) < min_instruction_chars:
            continue
        if len(response) < min_response_chars:
            continue

        expected = first_snippet(response, min_expected_chars, max_expected_chars)
        if len(expected) < min_expected_chars:
            continue

        eval_items.append(
            {
                "query": instruction,
                "expected_substring": expected,
            }
        )

        if len(eval_items) >= max_items:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eval_items, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "sft_path": str(sft_path),
        "sft_split_role": "eval_holdout",
        "out_path": str(out_path),
        "items": len(eval_items),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build auto eval QA file from holdout SFT split."
    )
    parser.add_argument("--sft-path", default="data/processed_v2_clean/splits/sft_eval_holdout.jsonl")
    parser.add_argument("--out", default="data/processed_v2_clean/eval_auto_qa.json")
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--min-instruction-chars", type=int, default=20)
    parser.add_argument("--min-response-chars", type=int, default=80)
    parser.add_argument("--min-expected-chars", type=int, default=30)
    parser.add_argument("--max-expected-chars", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    summary = build_eval(
        sft_path=Path(args.sft_path),
        out_path=Path(args.out),
        max_items=args.max_items,
        min_instruction_chars=args.min_instruction_chars,
        min_response_chars=args.min_response_chars,
        min_expected_chars=args.min_expected_chars,
        max_expected_chars=args.max_expected_chars,
        seed=args.seed,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
