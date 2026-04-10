import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_text(text)))


def _jaccard(a: str, b: str) -> float:
    left = _tokenize(a)
    right = _tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def _preview(text: str, limit: int = 140) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _rag_merged_text(row: dict[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    return f"{title}\n\n{text}".strip() if title else text


def _sft_combined_text(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    response = str(row.get("response", "")).strip()
    return f"{instruction}\n\n{response}".strip()


def detect_sft_eval_leakage(
    sft_rows: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    *,
    max_examples: int = 10,
    similarity_threshold: float = 0.92,
) -> tuple[dict[str, Any], list[str]]:
    examples: list[dict[str, Any]] = []
    errors: list[str] = []
    reason_counts: dict[str, int] = {
        "exact_query_match": 0,
        "expected_in_response": 0,
        "near_duplicate_pair": 0,
    }

    for eval_index, item in enumerate(eval_items, start=1):
        query = str(item.get("query", "")).strip()
        expected = str(item.get("expected_substring", "")).strip()
        query_norm = normalize_text(query)
        expected_norm = normalize_text(expected)
        combo = f"{query}\n\n{expected}".strip()
        matched_reasons: set[str] = set()
        first_row: dict[str, Any] | None = None

        for row in sft_rows:
            instruction = str(row.get("instruction", "")).strip()
            response = str(row.get("response", "")).strip()
            instruction_norm = normalize_text(instruction)
            response_norm = normalize_text(response)
            combined = _sft_combined_text(row)

            if query_norm and query_norm == instruction_norm:
                matched_reasons.add("exact_query_match")
            if expected_norm and expected_norm in response_norm:
                matched_reasons.add("expected_in_response")
            if _sequence_ratio(combo, combined) >= similarity_threshold:
                matched_reasons.add("near_duplicate_pair")

            if matched_reasons:
                first_row = row
                break

        if matched_reasons:
            for reason in matched_reasons:
                reason_counts[reason] += 1
                errors.append(
                    f"Eval item #{eval_index} overlaps with SFT split via {reason}"
                )
            if len(examples) < max_examples and first_row is not None:
                examples.append(
                    {
                        "eval_index": eval_index,
                        "reasons": sorted(matched_reasons),
                        "query": query,
                        "expected_substring": expected,
                        "source": str(first_row.get("source", "")),
                        "sft_id": str(first_row.get("id", "")),
                        "instruction_preview": _preview(first_row.get("instruction", "")),
                        "response_preview": _preview(first_row.get("response", "")),
                    }
                )

    return {
        "matches": sum(reason_counts.values()),
        "items_checked": len(eval_items),
        "reason_counts": reason_counts,
        "examples": examples,
    }, errors


def detect_rag_eval_leakage(
    rag_rows: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    *,
    max_examples: int = 10,
    similarity_threshold: float = 0.88,
    token_overlap_threshold: float = 0.55,
) -> tuple[dict[str, Any], list[str]]:
    examples: list[dict[str, Any]] = []
    errors: list[str] = []
    reason_counts: dict[str, int] = {
        "expected_in_rag_text": 0,
        "near_duplicate_document": 0,
    }

    for eval_index, item in enumerate(eval_items, start=1):
        query = str(item.get("query", "")).strip()
        expected = str(item.get("expected_substring", "")).strip()
        expected_norm = normalize_text(expected)
        combo = f"{query}\n\n{expected}".strip()
        matched_reasons: set[str] = set()
        first_row: dict[str, Any] | None = None

        for row in rag_rows:
            merged = _rag_merged_text(row)
            merged_norm = normalize_text(merged)
            title = str(row.get("title", "")).strip()

            query_matches_document = (
                _jaccard(query, merged) >= token_overlap_threshold
                or _sequence_ratio(query, title) >= 0.78
                or _sequence_ratio(query, merged) >= similarity_threshold
            )

            if expected_norm and expected_norm in merged_norm and query_matches_document:
                matched_reasons.add("expected_in_rag_text")
            if _sequence_ratio(combo, merged) >= similarity_threshold:
                matched_reasons.add("near_duplicate_document")

            if matched_reasons:
                first_row = row
                break

        if matched_reasons:
            for reason in matched_reasons:
                reason_counts[reason] += 1
                errors.append(
                    f"Eval item #{eval_index} overlaps with RAG corpus via {reason}"
                )
            if len(examples) < max_examples and first_row is not None:
                examples.append(
                    {
                        "eval_index": eval_index,
                        "reasons": sorted(matched_reasons),
                        "query": query,
                        "expected_substring": expected,
                        "source": str(first_row.get("source", "")),
                        "rag_id": str(first_row.get("id", "")),
                        "title": str(first_row.get("title", "")),
                        "text_preview": _preview(first_row.get("text", "")),
                    }
                )

    return {
        "matches": sum(reason_counts.values()),
        "items_checked": len(eval_items),
        "reason_counts": reason_counts,
        "examples": examples,
    }, errors


def detect_eval_eval_leakage(
    left_name: str,
    left_items: list[dict[str, Any]],
    right_name: str,
    right_items: list[dict[str, Any]],
    *,
    max_examples: int = 10,
    similarity_threshold: float = 0.94,
) -> tuple[dict[str, Any], list[str]]:
    examples: list[dict[str, Any]] = []
    errors: list[str] = []
    reason_counts = {
        "duplicate_query": 0,
        "near_duplicate_eval_item": 0,
    }

    for left_index, left_item in enumerate(left_items, start=1):
        left_query = str(left_item.get("query", "")).strip()
        left_expected = str(left_item.get("expected_substring", "")).strip()
        left_query_norm = normalize_text(left_query)
        left_combo = f"{left_query}\n\n{left_expected}".strip()
        matched_reasons: set[str] = set()
        right_match: dict[str, Any] | None = None
        right_match_index = -1

        for right_index, right_item in enumerate(right_items, start=1):
            right_query = str(right_item.get("query", "")).strip()
            right_expected = str(right_item.get("expected_substring", "")).strip()
            right_query_norm = normalize_text(right_query)
            right_combo = f"{right_query}\n\n{right_expected}".strip()

            if left_query_norm and left_query_norm == right_query_norm:
                matched_reasons.add("duplicate_query")
            if _sequence_ratio(left_combo, right_combo) >= similarity_threshold:
                matched_reasons.add("near_duplicate_eval_item")

            if matched_reasons:
                right_match = right_item
                right_match_index = right_index
                break

        if matched_reasons:
            for reason in matched_reasons:
                reason_counts[reason] += 1
                errors.append(
                    f"{left_name} item #{left_index} overlaps with {right_name} item #{right_match_index} via {reason}"
                )
            if len(examples) < max_examples and right_match is not None:
                examples.append(
                    {
                        "left_index": left_index,
                        "right_index": right_match_index,
                        "reasons": sorted(matched_reasons),
                        "left_query": left_query,
                        "right_query": str(right_match.get("query", "")),
                        "left_expected_preview": _preview(left_expected),
                        "right_expected_preview": _preview(right_match.get("expected_substring", "")),
                    }
                )

    return {
        "matches": sum(reason_counts.values()),
        "items_checked": len(left_items),
        "reason_counts": reason_counts,
        "examples": examples,
    }, errors


def validate_split_integrity(
    *,
    rag_rows: list[dict[str, Any]] | None = None,
    sft_rows: list[dict[str, Any]] | None = None,
    auto_eval_items: list[dict[str, Any]] | None = None,
    manual_eval_items: list[dict[str, Any]] | None = None,
    max_examples: int = 10,
) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    errors: list[str] = []

    rag_rows = rag_rows or []
    sft_rows = sft_rows or []
    auto_eval_items = auto_eval_items or []
    manual_eval_items = manual_eval_items or []

    if sft_rows and auto_eval_items:
        checks["sft_to_auto_eval"], check_errors = detect_sft_eval_leakage(
            sft_rows,
            auto_eval_items,
            max_examples=max_examples,
        )
        errors.extend(check_errors)

    if rag_rows and auto_eval_items:
        checks["rag_to_auto_eval"], check_errors = detect_rag_eval_leakage(
            rag_rows,
            auto_eval_items,
            max_examples=max_examples,
        )
        errors.extend(check_errors)

    if sft_rows and manual_eval_items:
        checks["sft_to_manual_eval"], check_errors = detect_sft_eval_leakage(
            sft_rows,
            manual_eval_items,
            max_examples=max_examples,
        )
        errors.extend(check_errors)

    if rag_rows and manual_eval_items:
        checks["rag_to_manual_eval"], check_errors = detect_rag_eval_leakage(
            rag_rows,
            manual_eval_items,
            max_examples=max_examples,
        )
        errors.extend(check_errors)

    if auto_eval_items and manual_eval_items:
        checks["auto_eval_to_manual_eval"], check_errors = detect_eval_eval_leakage(
            "auto_eval",
            auto_eval_items,
            "manual_eval",
            manual_eval_items,
            max_examples=max_examples,
        )
        errors.extend(check_errors)

    summary = {
        "checks": checks,
        "error_count": len(errors),
        "rag_rows": len(rag_rows),
        "sft_rows": len(sft_rows),
        "auto_eval_items": len(auto_eval_items),
        "manual_eval_items": len(manual_eval_items),
    }
    return summary, errors
