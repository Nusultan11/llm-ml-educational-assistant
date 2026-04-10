import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_NOISE_PATTERN = r"(?:http[s]?://|<[^>]+>|\bN/A\b|\blorem ipsum\b)"
MISSING_SOURCE_BUCKET = "__missing_source__"
DEFAULT_AUDIT_SAMPLE_LIMIT = 10
QUALITY_BUCKET_THRESHOLDS = {
    "low": 0.45,
    "medium": 0.72,
}
CODE_PATTERN = re.compile(r"(?:`[^`]+`|```|\bdef\s+\w+\(|\bclass\s+\w+\b|\bprint\(|\bimport\s+\w+)", re.IGNORECASE)
OPENASSISTANT_WEAK_REPLY_PATTERN = re.compile(
    r"(?:\bthanks?\b|\bthank you\b|\bglad to help\b|\bhappy to help\b|\bsure thing\b|\blet me know\b)",
    re.IGNORECASE,
)
WEAK_SFT_RESPONSES = {
    "yes",
    "no",
    "ok",
    "okay",
    "sure",
    "maybe",
    "n/a",
    "unknown",
    "i don't know",
    "idk",
}

RAG_QUALITY_RULES = {
    "required_fields": ["source", "text"],
    "reject_reasons": {
        "missing_source": "row has no usable source field",
        "empty_text": "text field is empty after normalization",
        "too_short_for_rag": "title + text is too short for retrieval use",
        "weak_title_text_pair": "record does not provide enough semantic context for retrieval",
        "noise": "row contains obvious noise markers such as links or lorem ipsum",
        "high_non_ascii_ratio": "row is dominated by non-ascii noise for this pipeline",
        "arxiv_missing_title": "arxiv rows should include a meaningful paper title",
        "stackoverflow_low_signal_answer": "stackoverflow rows should contain more explanation than a tiny code snippet",
        "duplicate": "normalized retrieval document duplicates an earlier record",
    },
}

SFT_QUALITY_RULES = {
    "required_fields": ["source", "instruction", "response"],
    "reject_reasons": {
        "missing_source": "row has no usable source field",
        "empty_instruction": "instruction field is empty after normalization",
        "empty_response": "response field is empty after normalization",
        "instruction_too_short": "instruction is too short to supervise a useful behavior",
        "response_too_short": "response is too short to teach the model anything useful",
        "weak_response": "response is present but too weak or generic to be a good target",
        "invalid_instruction_response_pair": "instruction and response do not form a meaningful pair",
        "noise": "row contains obvious noise markers such as links or lorem ipsum",
        "high_non_ascii_ratio": "row is dominated by non-ascii noise for this pipeline",
        "openassistant_weak_assistant_reply": "openassistant replies should be more substantial than short conversational acknowledgements",
        "stackoverflow_low_signal_answer": "stackoverflow supervision pairs should include more explanation than a terse code hint",
        "duplicate": "normalized instruction-response pair duplicates an earlier record",
    },
}

SOURCE_SPECIFIC_RULES = {
    "rag": {
        "arxiv": {
            "description": "ArXiv rows should look like document records with a meaningful title plus abstract-style text.",
            "rules": [
                {
                    "reject_reason": "arxiv_missing_title",
                    "description": "reject when title is empty or too short for a paper-like source",
                }
            ],
        },
        "stackoverflow": {
            "description": "StackOverflow retrieval rows should not be mostly code or snippet fragments without enough explanation.",
            "rules": [
                {
                    "reject_reason": "stackoverflow_low_signal_answer",
                    "description": "reject short code-heavy answers that are too thin for retrieval",
                }
            ],
        },
    },
    "sft": {
        "openassistant": {
            "description": "OpenAssistant assistant replies should be more substantive than brief conversational acknowledgements.",
            "rules": [
                {
                    "reject_reason": "openassistant_weak_assistant_reply",
                    "description": "reject short, polite assistant-style replies that do not teach a useful behavior",
                }
            ],
        },
        "stackoverflow": {
            "description": "StackOverflow supervision pairs should include explanatory answers, not just a terse code hint.",
            "rules": [
                {
                    "reject_reason": "stackoverflow_low_signal_answer",
                    "description": "reject short code-heavy answers that are too thin for supervision",
                }
            ],
        },
    },
}

QUALITY_SCORING_RULES = {
    "rag": {
        "description": "Scores kept retrieval rows by semantic richness, document structure, and source-specific fit.",
        "buckets": {
            "low": f"score < {QUALITY_BUCKET_THRESHOLDS['low']:.2f}",
            "medium": f"{QUALITY_BUCKET_THRESHOLDS['low']:.2f} <= score < {QUALITY_BUCKET_THRESHOLDS['medium']:.2f}",
            "high": f"score >= {QUALITY_BUCKET_THRESHOLDS['medium']:.2f}",
        },
    },
    "sft": {
        "description": "Scores kept supervision pairs by instructional clarity, response depth, and practical usefulness.",
        "buckets": {
            "low": f"score < {QUALITY_BUCKET_THRESHOLDS['low']:.2f}",
            "medium": f"{QUALITY_BUCKET_THRESHOLDS['low']:.2f} <= score < {QUALITY_BUCKET_THRESHOLDS['medium']:.2f}",
            "high": f"score >= {QUALITY_BUCKET_THRESHOLDS['medium']:.2f}",
        },
    },
}


@dataclass(frozen=True)
class RowValidationResult:
    valid: bool
    reason: str = ""


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


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _source_bucket(row: dict[str, Any]) -> str:
    source = normalize_text(row.get("source", ""))
    return source or MISSING_SOURCE_BUCKET


def _init_source_profile() -> dict[str, Any]:
    return {
        "before": 0,
        "after": 0,
        "removed": 0,
        "drop_reasons": Counter(),
        "quality_buckets": Counter(),
        "quality_score_sum": 0.0,
    }


def _finalize_source_profiles(profiles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for source in sorted(profiles):
        profile = profiles[source]
        finalized[source] = {
            "before": profile["before"],
            "after": profile["after"],
            "removed": profile["before"] - profile["after"],
            "drop_reasons": dict(profile["drop_reasons"]),
            "quality_buckets": dict(profile["quality_buckets"]),
            "avg_quality_score": round(
                profile["quality_score_sum"] / profile["after"], 3
            ) if profile["after"] else 0.0,
        }
    return finalized


def _source_specific_rag_reason(source: str, title: str, text: str) -> str | None:
    source = normalize_text(source).lower()

    if source == "arxiv":
        if len(normalize_text(title)) < 8:
            return "arxiv_missing_title"

    if source == "stackoverflow":
        if _word_count(text) < 12 and CODE_PATTERN.search(text):
            return "stackoverflow_low_signal_answer"

    return None


def _source_specific_sft_reason(source: str, instruction: str, response: str) -> str | None:
    source = normalize_text(source).lower()

    if source == "openassistant":
        if _word_count(response) < 6 and OPENASSISTANT_WEAK_REPLY_PATTERN.search(response):
            return "openassistant_weak_assistant_reply"

    if source == "stackoverflow":
        if _word_count(response) < 8 and CODE_PATTERN.search(response):
            return "stackoverflow_low_signal_answer"

    return None


def _safe_bucket_name(value: str) -> str:
    normalized = normalize_text(value).lower()
    sanitized = re.sub(r"[^a-z0-9_-]+", "_", normalized).strip("_")
    return sanitized or "unknown"


def _preview_text(text: str, *, limit: int = 180) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _quality_bucket(score: float) -> str:
    if score < QUALITY_BUCKET_THRESHOLDS["low"]:
        return "low"
    if score < QUALITY_BUCKET_THRESHOLDS["medium"]:
        return "medium"
    return "high"


def score_rag_row(row: dict[str, Any]) -> tuple[float, str]:
    source = normalize_text(row.get("source", "")).lower()
    title = normalize_text(row.get("title", ""))
    text = normalize_text(row.get("text", ""))
    merged = f"{title}\n\n{text}" if title else text

    score = 0.42
    if title:
        score += min(len(title) / 120, 1.0) * 0.14
    score += min(len(text) / 900, 1.0) * 0.22
    score += min(_word_count(text) / 160, 1.0) * 0.16
    if any(mark in merged for mark in [". ", "? ", "! ", ": "]):
        score += 0.06
    if "\n" in merged:
        score += 0.04

    if source == "arxiv" and title and len(text) >= 200:
        score += 0.10
    if source == "stackoverflow" and not CODE_PATTERN.search(text):
        score += 0.06
    if source in {"openassistant", "dolly"} and len(text) >= 140:
        score += 0.05

    score = _clamp_score(score)
    return score, _quality_bucket(score)


def score_sft_row(row: dict[str, Any]) -> tuple[float, str]:
    source = normalize_text(row.get("source", "")).lower()
    instruction = normalize_text(row.get("instruction", ""))
    response = normalize_text(row.get("response", ""))
    combined = f"{instruction}\n\n{response}".strip()

    score = 0.38
    score += min(len(instruction) / 180, 1.0) * 0.16
    score += min(len(response) / 1200, 1.0) * 0.26
    score += min(_word_count(response) / 180, 1.0) * 0.16
    if any(marker in response.lower() for marker in ["for example", "example", "because", "steps", "first", "second"]):
        score += 0.08
    if any(marker in response for marker in [". ", ": ", "\n"]):
        score += 0.06

    if source == "openassistant" and _word_count(response) >= 30:
        score += 0.06
    if source == "stackoverflow" and not CODE_PATTERN.search(response):
        score += 0.05
    if source == "dolly" and len(instruction) >= 40 and len(response) >= 160:
        score += 0.06

    if OPENASSISTANT_WEAK_REPLY_PATTERN.search(response):
        score -= 0.08

    score = _clamp_score(score)
    return score, _quality_bucket(score)


def _build_audit_sample(
    row: dict[str, Any],
    *,
    dataset_type: str,
    status: str,
    source_bucket: str,
    reject_reason: str | None = None,
) -> dict[str, Any]:
    sample = {
        "dataset_type": dataset_type,
        "status": status,
        "reject_reason": reject_reason,
        "id": str(row.get("id", "")),
        "source": source_bucket,
    }
    quality_score = row.get("quality_score")
    quality_bucket = row.get("quality_bucket")
    if quality_score is not None:
        sample["quality_score"] = quality_score
    if quality_bucket is not None:
        sample["quality_bucket"] = quality_bucket

    if dataset_type == "rag":
        title = normalize_text(row.get("title", ""))
        text = normalize_text(row.get("text", ""))
        merged = f"{title}\n\n{text}" if title else text
        sample.update(
            {
                "title": title,
                "text": text,
                "preview": _preview_text(merged),
                "normalized_length": len(merged),
            }
        )
    else:
        instruction = normalize_text(row.get("instruction", ""))
        response = normalize_text(row.get("response", ""))
        combined = f"{instruction} {response}".strip()
        sample.update(
            {
                "instruction": instruction,
                "response": response,
                "preview": _preview_text(combined),
                "normalized_length": len(combined),
            }
        )

    return sample


def _record_audit_sample(
    samples: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    *,
    dataset_type: str,
    category: str,
    source_bucket: str,
    bucket_name: str,
    sample: dict[str, Any],
    sample_limit: int,
) -> None:
    key = (dataset_type, category, source_bucket, bucket_name)
    bucket = samples.setdefault(key, [])
    if len(bucket) < sample_limit:
        bucket.append(sample)


def write_audit_samples(
    audit_dir: Path,
    *,
    samples: dict[tuple[str, str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"files": []}

    for dataset_type, category, source_bucket, bucket_name in sorted(samples):
        file_dir = audit_dir / dataset_type / category
        source_slug = _safe_bucket_name(source_bucket)
        bucket_slug = _safe_bucket_name(bucket_name)
        filename = f"{source_slug}__{bucket_slug}.jsonl"
        path = file_dir / filename
        write_jsonl(path, samples[(dataset_type, category, source_bucket, bucket_name)])
        manifest["files"].append(
            {
                "dataset_type": dataset_type,
                "category": category,
                "source": source_bucket,
                "bucket": bucket_name,
                "path": str(path),
                "sample_count": len(samples[(dataset_type, category, source_bucket, bucket_name)]),
            }
        )

    manifest["file_count"] = len(manifest["files"])
    return manifest


def validate_rag_row(
    row: dict[str, Any],
    *,
    min_chars: int,
    max_non_ascii: float,
    noise_pattern: re.Pattern[str],
) -> RowValidationResult:
    source = normalize_text(row.get("source", ""))
    title = normalize_text(row.get("title", ""))
    text = normalize_text(row.get("text", ""))
    merged = f"{title}\n\n{text}" if title else text

    if not source:
        return RowValidationResult(False, "missing_source")
    if not text:
        return RowValidationResult(False, "empty_text")
    if len(merged) < min_chars:
        return RowValidationResult(False, "too_short_for_rag")
    if _word_count(text) < 6 or (not title and _word_count(text) < 12):
        return RowValidationResult(False, "weak_title_text_pair")
    if noise_pattern.search(merged):
        return RowValidationResult(False, "noise")
    if non_ascii_ratio(merged) > max_non_ascii:
        return RowValidationResult(False, "high_non_ascii_ratio")
    source_specific_reason = _source_specific_rag_reason(source, title, text)
    if source_specific_reason:
        return RowValidationResult(False, source_specific_reason)
    return RowValidationResult(True)


def validate_sft_row(
    row: dict[str, Any],
    *,
    min_instruction_chars: int,
    min_response_chars: int,
    max_non_ascii: float,
    noise_pattern: re.Pattern[str],
) -> RowValidationResult:
    source = normalize_text(row.get("source", ""))
    instruction = normalize_text(row.get("instruction", ""))
    response = normalize_text(row.get("response", ""))
    combined = f"{instruction} {response}".strip()

    if not source:
        return RowValidationResult(False, "missing_source")
    if not instruction:
        return RowValidationResult(False, "empty_instruction")
    if not response:
        return RowValidationResult(False, "empty_response")
    if len(instruction) < min_instruction_chars:
        return RowValidationResult(False, "instruction_too_short")
    if len(response) < min_response_chars:
        return RowValidationResult(False, "response_too_short")
    if normalize_text(response).lower() in WEAK_SFT_RESPONSES or _word_count(response) < 4:
        return RowValidationResult(False, "weak_response")
    if normalize_text(instruction).lower() == normalize_text(response).lower():
        return RowValidationResult(False, "invalid_instruction_response_pair")
    if response.lower().startswith("context:"):
        return RowValidationResult(False, "invalid_instruction_response_pair")
    if noise_pattern.search(combined):
        return RowValidationResult(False, "noise")
    if non_ascii_ratio(combined) > max_non_ascii:
        return RowValidationResult(False, "high_non_ascii_ratio")
    source_specific_reason = _source_specific_sft_reason(source, instruction, response)
    if source_specific_reason:
        return RowValidationResult(False, source_specific_reason)
    return RowValidationResult(True)


def clean_rag_rows(
    rows: list[dict[str, Any]],
    *,
    min_chars: int,
    max_non_ascii: float,
    noise_pattern: re.Pattern[str],
    audit_samples: dict[tuple[str, str, str, str], list[dict[str, Any]]] | None = None,
    audit_sample_limit: int = DEFAULT_AUDIT_SAMPLE_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, Any]]]:
    cleaned: list[dict[str, Any]] = []
    dropped = Counter()
    seen: set[str] = set()
    per_source: dict[str, dict[str, Any]] = {}

    for row in rows:
        source_bucket = _source_bucket(row)
        profile = per_source.setdefault(source_bucket, _init_source_profile())
        profile["before"] += 1

        title = normalize_text(row.get("title", ""))
        text = normalize_text(row.get("text", ""))
        merged = f"{title}\n\n{text}" if title else text

        validation = validate_rag_row(
            row,
            min_chars=min_chars,
            max_non_ascii=max_non_ascii,
            noise_pattern=noise_pattern,
        )
        if not validation.valid:
            dropped[validation.reason] += 1
            profile["drop_reasons"][validation.reason] += 1
            if audit_samples is not None:
                _record_audit_sample(
                    audit_samples,
                    dataset_type="rag",
                    category="rejected",
                    source_bucket=source_bucket,
                    bucket_name=validation.reason,
                    sample=_build_audit_sample(
                        row,
                        dataset_type="rag",
                        status="rejected",
                        source_bucket=source_bucket,
                        reject_reason=validation.reason,
                    ),
                    sample_limit=audit_sample_limit,
                )
            continue

        key = normalize_text(merged).lower()
        if key in seen:
            dropped["duplicate"] += 1
            profile["drop_reasons"]["duplicate"] += 1
            if audit_samples is not None:
                _record_audit_sample(
                    audit_samples,
                    dataset_type="rag",
                    category="rejected",
                    source_bucket=source_bucket,
                    bucket_name="duplicate",
                    sample=_build_audit_sample(
                        row,
                        dataset_type="rag",
                        status="rejected",
                        source_bucket=source_bucket,
                        reject_reason="duplicate",
                    ),
                    sample_limit=audit_sample_limit,
                )
            continue
        seen.add(key)

        tags = row.get("tags", ["ml_assistant"])
        if not isinstance(tags, list):
            tags = ["ml_assistant"]

        quality_score, quality_bucket = score_rag_row(row)
        profile["after"] += 1
        profile["quality_buckets"][quality_bucket] += 1
        profile["quality_score_sum"] += quality_score
        kept_row = {
            "id": str(row.get("id", f"rag-{len(cleaned)}")),
            "source": normalize_text(row.get("source", "")),
            "title": title,
            "text": text,
            "tags": tags,
            "quality_score": quality_score,
            "quality_bucket": quality_bucket,
        }
        if audit_samples is not None:
            _record_audit_sample(
                audit_samples,
                dataset_type="rag",
                category="kept",
                source_bucket=source_bucket,
                bucket_name=f"kept_{quality_bucket}_quality",
                sample=_build_audit_sample(
                    kept_row,
                    dataset_type="rag",
                    status="kept",
                    source_bucket=source_bucket,
                    reject_reason=None,
                ),
                sample_limit=audit_sample_limit,
            )
        cleaned.append(kept_row)

    return cleaned, dict(dropped), _finalize_source_profiles(per_source)


def clean_sft_rows(
    rows: list[dict[str, Any]],
    *,
    min_instruction_chars: int,
    min_response_chars: int,
    max_non_ascii: float,
    noise_pattern: re.Pattern[str],
    audit_samples: dict[tuple[str, str, str, str], list[dict[str, Any]]] | None = None,
    audit_sample_limit: int = DEFAULT_AUDIT_SAMPLE_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, Any]]]:
    cleaned: list[dict[str, Any]] = []
    dropped = Counter()
    seen: set[str] = set()
    per_source: dict[str, dict[str, Any]] = {}

    for row in rows:
        source_bucket = _source_bucket(row)
        profile = per_source.setdefault(source_bucket, _init_source_profile())
        profile["before"] += 1

        instruction = normalize_text(row.get("instruction", ""))
        response = normalize_text(row.get("response", ""))

        validation = validate_sft_row(
            row,
            min_instruction_chars=min_instruction_chars,
            min_response_chars=min_response_chars,
            max_non_ascii=max_non_ascii,
            noise_pattern=noise_pattern,
        )
        if not validation.valid:
            dropped[validation.reason] += 1
            profile["drop_reasons"][validation.reason] += 1
            if audit_samples is not None:
                _record_audit_sample(
                    audit_samples,
                    dataset_type="sft",
                    category="rejected",
                    source_bucket=source_bucket,
                    bucket_name=validation.reason,
                    sample=_build_audit_sample(
                        row,
                        dataset_type="sft",
                        status="rejected",
                        source_bucket=source_bucket,
                        reject_reason=validation.reason,
                    ),
                    sample_limit=audit_sample_limit,
                )
            continue

        key = f"{normalize_text(instruction).lower()}|||{normalize_text(response).lower()}"
        if key in seen:
            dropped["duplicate"] += 1
            profile["drop_reasons"]["duplicate"] += 1
            if audit_samples is not None:
                _record_audit_sample(
                    audit_samples,
                    dataset_type="sft",
                    category="rejected",
                    source_bucket=source_bucket,
                    bucket_name="duplicate",
                    sample=_build_audit_sample(
                        row,
                        dataset_type="sft",
                        status="rejected",
                        source_bucket=source_bucket,
                        reject_reason="duplicate",
                    ),
                    sample_limit=audit_sample_limit,
                )
            continue
        seen.add(key)

        quality_score, quality_bucket = score_sft_row(row)
        profile["after"] += 1
        profile["quality_buckets"][quality_bucket] += 1
        profile["quality_score_sum"] += quality_score
        kept_row = {
            "id": str(row.get("id", f"sft-{len(cleaned)}")),
            "source": normalize_text(row.get("source", "")),
            "instruction": instruction,
            "response": response,
            "quality_score": quality_score,
            "quality_bucket": quality_bucket,
        }
        if audit_samples is not None:
            _record_audit_sample(
                audit_samples,
                dataset_type="sft",
                category="kept",
                source_bucket=source_bucket,
                bucket_name=f"kept_{quality_bucket}_quality",
                sample=_build_audit_sample(
                    kept_row,
                    dataset_type="sft",
                    status="kept",
                    source_bucket=source_bucket,
                    reject_reason=None,
                ),
                sample_limit=audit_sample_limit,
            )
        cleaned.append(kept_row)

    return cleaned, dict(dropped), _finalize_source_profiles(per_source)


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
    audit_dir: Path | None,
    min_rag_chars: int,
    min_instruction_chars: int,
    min_response_chars: int,
    max_non_ascii_ratio: float,
    noise_pattern: str,
    audit_sample_limit: int,
) -> dict[str, Any]:
    rag_in = in_dir / "rag_corpus.jsonl"
    sft_in = in_dir / "sft_instructions.jsonl"
    if not rag_in.exists() or not sft_in.exists():
        raise FileNotFoundError(f"Input files were not found in: {in_dir}")

    rag_rows = read_jsonl(rag_in)
    sft_rows = read_jsonl(sft_in)
    compiled_noise = re.compile(noise_pattern, re.IGNORECASE)
    audit_samples: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}

    cleaned_rag, rag_dropped, rag_per_source = clean_rag_rows(
        rag_rows,
        min_chars=min_rag_chars,
        max_non_ascii=max_non_ascii_ratio,
        noise_pattern=compiled_noise,
        audit_samples=audit_samples,
        audit_sample_limit=audit_sample_limit,
    )
    cleaned_sft, sft_dropped, sft_per_source = clean_sft_rows(
        sft_rows,
        min_instruction_chars=min_instruction_chars,
        min_response_chars=min_response_chars,
        max_non_ascii=max_non_ascii_ratio,
        noise_pattern=compiled_noise,
        audit_samples=audit_samples,
        audit_sample_limit=audit_sample_limit,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rag_out = out_dir / "rag_corpus.jsonl"
    sft_out = out_dir / "sft_instructions.jsonl"
    write_jsonl(rag_out, cleaned_rag)
    write_jsonl(sft_out, cleaned_sft)

    rag_docs_count = 0
    if rag_docs_dir is not None:
        rag_docs_count = write_rag_docs(cleaned_rag, rag_docs_dir)

    audit_manifest = None
    if audit_dir is not None:
        audit_manifest = write_audit_samples(audit_dir, samples=audit_samples)

    summary = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "rag_docs_dir": str(rag_docs_dir) if rag_docs_dir else None,
        "audit_dir": str(audit_dir) if audit_dir else None,
        "params": {
            "min_rag_chars": min_rag_chars,
            "min_instruction_chars": min_instruction_chars,
            "min_response_chars": min_response_chars,
            "max_non_ascii_ratio": max_non_ascii_ratio,
            "noise_pattern": noise_pattern,
            "audit_sample_limit": audit_sample_limit,
        },
        "quality_standards": {
            "rag": RAG_QUALITY_RULES,
            "sft": SFT_QUALITY_RULES,
        },
        "quality_scoring": QUALITY_SCORING_RULES,
        "source_specific_rules": SOURCE_SPECIFIC_RULES,
        "rag": {
            "before": len(rag_rows),
            "after": len(cleaned_rag),
            "removed": len(rag_rows) - len(cleaned_rag),
            "drop_reasons": rag_dropped,
            "per_source": rag_per_source,
            "quality_buckets": dict(Counter(row["quality_bucket"] for row in cleaned_rag)),
            "avg_quality_score": round(
                sum(float(row["quality_score"]) for row in cleaned_rag) / len(cleaned_rag), 3
            ) if cleaned_rag else 0.0,
        },
        "sft": {
            "before": len(sft_rows),
            "after": len(cleaned_sft),
            "removed": len(sft_rows) - len(cleaned_sft),
            "drop_reasons": sft_dropped,
            "per_source": sft_per_source,
            "quality_buckets": dict(Counter(row["quality_bucket"] for row in cleaned_sft)),
            "avg_quality_score": round(
                sum(float(row["quality_score"]) for row in cleaned_sft) / len(cleaned_sft), 3
            ) if cleaned_sft else 0.0,
        },
        "rag_docs_count": rag_docs_count,
        "audit_sampling": audit_manifest,
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
    parser.add_argument("--audit-dir", default="reports/data_audit")
    parser.add_argument("--min-rag-chars", type=int, default=80)
    parser.add_argument("--min-instruction-chars", type=int, default=10)
    parser.add_argument("--min-response-chars", type=int, default=40)
    parser.add_argument("--max-non-ascii-ratio", type=float, default=0.30)
    parser.add_argument("--noise-pattern", default=DEFAULT_NOISE_PATTERN)
    parser.add_argument("--audit-sample-limit", type=int, default=DEFAULT_AUDIT_SAMPLE_LIMIT)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rag_docs_dir = Path(args.rag_docs_dir) if args.rag_docs_dir else None
    audit_dir = Path(args.audit_dir) if args.audit_dir else None
    summary = clean_processed_datasets(
        in_dir=Path(args.in_dir),
        out_dir=Path(args.out_dir),
        rag_docs_dir=rag_docs_dir,
        audit_dir=audit_dir,
        min_rag_chars=args.min_rag_chars,
        min_instruction_chars=args.min_instruction_chars,
        min_response_chars=args.min_response_chars,
        max_non_ascii_ratio=args.max_non_ascii_ratio,
        noise_pattern=args.noise_pattern,
        audit_sample_limit=args.audit_sample_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
