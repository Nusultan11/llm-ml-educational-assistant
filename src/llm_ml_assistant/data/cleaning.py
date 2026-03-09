import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_NOISE_PATTERN = r"(?:http[s]?://|<[^>]+>|\bN/A\b|\blorem ipsum\b)"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii_chars = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii_chars / max(len(text), 1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_rag_rows(
    rows: list[dict[str, Any]],
    *,
    min_chars: int,
    max_non_ascii: float,
    noise_pattern: re.Pattern[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cleaned: list[dict[str, Any]] = []
    dropped = Counter()
    seen: set[str] = set()

    for row in rows:
        title = normalize_text(row.get("title", ""))
        text = normalize_text(row.get("text", ""))
        merged = f"{title}\n\n{text}" if title else text

        if not text:
            dropped["empty"] += 1
            continue
        if len(merged) < min_chars:
            dropped["too_short"] += 1
            continue
        if noise_pattern.search(merged):
            dropped["noise"] += 1
            continue
        if non_ascii_ratio(merged) > max_non_ascii:
            dropped["high_non_ascii_ratio"] += 1
            continue

        key = normalize_text(merged).lower()
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)

        tags = row.get("tags", ["ml_assistant"])
        if not isinstance(tags, list):
            tags = ["ml_assistant"]

        cleaned.append(
            {
                "id": str(row.get("id", f"rag-{len(cleaned)}")),
                "source": normalize_text(row.get("source", "")) or "unknown",
                "title": title,
                "text": text,
                "tags": tags,
            }
        )

    return cleaned, dict(dropped)


def clean_sft_rows(
    rows: list[dict[str, Any]],
    *,
    min_instruction_chars: int,
    min_response_chars: int,
    max_non_ascii: float,
    noise_pattern: re.Pattern[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cleaned: list[dict[str, Any]] = []
    dropped = Counter()
    seen: set[str] = set()

    for row in rows:
        instruction = normalize_text(row.get("instruction", ""))
        response = normalize_text(row.get("response", ""))
        combined = f"{instruction} {response}".strip()

        if not instruction or not response:
            dropped["empty"] += 1
            continue
        if len(instruction) < min_instruction_chars:
            dropped["instruction_too_short"] += 1
            continue
        if len(response) < min_response_chars:
            dropped["response_too_short"] += 1
            continue
        if noise_pattern.search(combined):
            dropped["noise"] += 1
            continue
        if non_ascii_ratio(combined) > max_non_ascii:
            dropped["high_non_ascii_ratio"] += 1
            continue

        key = f"{normalize_text(instruction).lower()}|||{normalize_text(response).lower()}"
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)

        cleaned.append(
            {
                "id": str(row.get("id", f"sft-{len(cleaned)}")),
                "source": normalize_text(row.get("source", "")) or "unknown",
                "instruction": instruction,
                "response": response,
            }
        )

    return cleaned, dict(dropped)


def write_rag_docs(rows: list[dict[str, Any]], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, row in enumerate(rows):
        title = normalize_text(row.get("title", ""))
        text = normalize_text(row.get("text", ""))
        if not text:
            continue
        merged = f"{title}\n\n{text}" if title else text
        (out_dir / f"doc_{i:07d}.txt").write_text(merged, encoding="utf-8")
        count += 1
    return count


def clean_processed_datasets(
    *,
    in_dir: Path,
    out_dir: Path,
    rag_docs_dir: Path | None,
    min_rag_chars: int,
    min_instruction_chars: int,
    min_response_chars: int,
    max_non_ascii_ratio: float,
    noise_pattern: str,
) -> dict[str, Any]:
    rag_in = in_dir / "rag_corpus.jsonl"
    sft_in = in_dir / "sft_instructions.jsonl"
    if not rag_in.exists() or not sft_in.exists():
        raise FileNotFoundError(f"Input files were not found in: {in_dir}")

    rag_rows = read_jsonl(rag_in)
    sft_rows = read_jsonl(sft_in)
    compiled_noise = re.compile(noise_pattern, re.IGNORECASE)

    cleaned_rag, rag_dropped = clean_rag_rows(
        rag_rows,
        min_chars=min_rag_chars,
        max_non_ascii=max_non_ascii_ratio,
        noise_pattern=compiled_noise,
    )
    cleaned_sft, sft_dropped = clean_sft_rows(
        sft_rows,
        min_instruction_chars=min_instruction_chars,
        min_response_chars=min_response_chars,
        max_non_ascii=max_non_ascii_ratio,
        noise_pattern=compiled_noise,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rag_out = out_dir / "rag_corpus.jsonl"
    sft_out = out_dir / "sft_instructions.jsonl"
    write_jsonl(rag_out, cleaned_rag)
    write_jsonl(sft_out, cleaned_sft)

    rag_docs_count = 0
    if rag_docs_dir is not None:
        rag_docs_count = write_rag_docs(cleaned_rag, rag_docs_dir)

    summary = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "rag_docs_dir": str(rag_docs_dir) if rag_docs_dir else None,
        "params": {
            "min_rag_chars": min_rag_chars,
            "min_instruction_chars": min_instruction_chars,
            "min_response_chars": min_response_chars,
            "max_non_ascii_ratio": max_non_ascii_ratio,
            "noise_pattern": noise_pattern,
        },
        "rag": {
            "before": len(rag_rows),
            "after": len(cleaned_rag),
            "removed": len(rag_rows) - len(cleaned_rag),
            "drop_reasons": rag_dropped,
        },
        "sft": {
            "before": len(sft_rows),
            "after": len(cleaned_sft),
            "removed": len(sft_rows) - len(cleaned_sft),
            "drop_reasons": sft_dropped,
        },
        "rag_docs_count": rag_docs_count,
    }

    (out_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean processed RAG/SFT datasets and write local v2 outputs.")
    parser.add_argument("--in-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed_v2_clean")
    parser.add_argument("--rag-docs-dir", default="data/rag_docs_v2_clean")
    parser.add_argument("--min-rag-chars", type=int, default=80)
    parser.add_argument("--min-instruction-chars", type=int, default=10)
    parser.add_argument("--min-response-chars", type=int, default=40)
    parser.add_argument("--max-non-ascii-ratio", type=float, default=0.30)
    parser.add_argument("--noise-pattern", default=DEFAULT_NOISE_PATTERN)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rag_docs_dir = Path(args.rag_docs_dir) if args.rag_docs_dir else None
    summary = clean_processed_datasets(
        in_dir=Path(args.in_dir),
        out_dir=Path(args.out_dir),
        rag_docs_dir=rag_docs_dir,
        min_rag_chars=args.min_rag_chars,
        min_instruction_chars=args.min_instruction_chars,
        min_response_chars=args.min_response_chars,
        max_non_ascii_ratio=args.max_non_ascii_ratio,
        noise_pattern=args.noise_pattern,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
