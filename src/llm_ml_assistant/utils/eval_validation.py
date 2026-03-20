import json
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def load_eval_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Eval file must contain a JSON list")
    return data


def validate_eval_items(
    items: list[dict],
    min_query_chars: int = 8,
    min_expected_chars: int = 8,
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    seen_queries: dict[str, int] = {}

    if not items:
        errors.append("Eval list is empty")

    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"Item #{i}: must be an object")
            continue

        query = str(item.get("query", "")).strip()
        expected = str(item.get("expected_substring", "")).strip()

        if len(query) < min_query_chars:
            errors.append(
                f"Item #{i}: query is too short ({len(query)} < {min_query_chars})"
            )

        if len(expected) < min_expected_chars:
            errors.append(
                f"Item #{i}: expected_substring is too short ({len(expected)} < {min_expected_chars})"
            )

        norm_query = normalize_text(query)
        if norm_query:
            if norm_query in seen_queries:
                prev = seen_queries[norm_query]
                errors.append(f"Item #{i}: duplicate query (first seen at item #{prev})")
            else:
                seen_queries[norm_query] = i

    summary = {
        "items": len(items),
        "unique_queries": len(seen_queries),
        "min_query_chars": min_query_chars,
        "min_expected_chars": min_expected_chars,
        "errors": len(errors),
    }

    return summary, errors
