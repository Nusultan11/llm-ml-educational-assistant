import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_QUALITY_POLICIES = {
    "train": {"high": 1.0, "medium": 0.8, "low": 0.35},
    "dev": {"high": 1.0, "medium": 0.7, "low": 0.3},
    "eval_holdout": {"high": 1.0, "medium": 0.85, "low": 0.5},
}
QUALITY_BUCKET_ORDER = ("high", "medium", "low")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _quality_bucket(row: dict[str, Any]) -> str:
    bucket = str(row.get("quality_bucket", "")).strip().lower()
    if bucket in QUALITY_BUCKET_ORDER:
        return bucket
    return "medium"


def _allocate_split_counts(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if count <= 0:
        return (0, 0, 0)

    raw = [count * ratio for ratio in ratios]
    base = [int(value) for value in raw]
    remainder = count - sum(base)

    order = sorted(
        range(len(ratios)),
        key=lambda idx: (raw[idx] - base[idx], ratios[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        base[idx] += 1

    train_count, dev_count, eval_count = base

    if count >= 3 and ratios[2] > 0 and eval_count == 0:
        if train_count > dev_count and train_count > 1:
            train_count -= 1
        elif dev_count > 0:
            dev_count -= 1
        else:
            train_count = max(train_count - 1, 0)
        eval_count += 1

    if count >= 2 and ratios[1] > 0 and dev_count == 0:
        if train_count > 1:
            train_count -= 1
            dev_count += 1

    total = train_count + dev_count + eval_count
    if total != count:
        train_count += count - total

    return train_count, dev_count, eval_count


def _select_rows_by_quality_policy(
    rows: list[dict[str, Any]],
    *,
    keep_rates: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_quality_bucket(row)].append(row)

    selected: list[dict[str, Any]] = []
    summary = {
        "before": len(rows),
        "after": 0,
        "per_bucket_before": {},
        "per_bucket_after": {},
        "keep_rates": keep_rates,
    }

    for bucket in QUALITY_BUCKET_ORDER:
        bucket_rows = grouped.get(bucket, [])
        rate = keep_rates.get(bucket, 1.0)
        keep_count = min(len(bucket_rows), int(round(len(bucket_rows) * rate)))
        kept_rows = bucket_rows[:keep_count]
        selected.extend(kept_rows)
        summary["per_bucket_before"][bucket] = len(bucket_rows)
        summary["per_bucket_after"][bucket] = len(kept_rows)

    summary["after"] = len(selected)
    return selected, summary


def split_sft_rows(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    eval_ratio: float = 0.1,
    seed: int = 42,
    quality_policies: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    ratio_sum = train_ratio + dev_ratio + eval_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    policies = quality_policies or DEFAULT_QUALITY_POLICIES
    for split_name in ("train", "dev", "eval_holdout"):
        if split_name not in policies:
            raise ValueError(f"Missing quality policy for split: {split_name}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source", "")).strip() or "__missing_source__"
        grouped[(source, _quality_bucket(row))].append(row)

    rng = random.Random(seed)
    raw_split_rows = {
        "train": [],
        "dev": [],
        "eval_holdout": [],
    }
    per_source: dict[str, Any] = defaultdict(
        lambda: {
            "total": 0,
            "train_raw": 0,
            "dev_raw": 0,
            "eval_holdout_raw": 0,
            "train": 0,
            "dev": 0,
            "eval_holdout": 0,
            "quality_buckets": Counter(),
        }
    )

    for (source, quality_bucket), source_rows in sorted(grouped.items()):
        source_rows = list(source_rows)
        rng.shuffle(source_rows)
        train_count, dev_count, eval_count = _allocate_split_counts(
            len(source_rows),
            (train_ratio, dev_ratio, eval_ratio),
        )

        train_rows = source_rows[:train_count]
        dev_rows = source_rows[train_count : train_count + dev_count]
        eval_rows = source_rows[train_count + dev_count : train_count + dev_count + eval_count]

        raw_split_rows["train"].extend(train_rows)
        raw_split_rows["dev"].extend(dev_rows)
        raw_split_rows["eval_holdout"].extend(eval_rows)

        source_profile = per_source[source]
        source_profile["total"] += len(source_rows)
        source_profile["train_raw"] += len(train_rows)
        source_profile["dev_raw"] += len(dev_rows)
        source_profile["eval_holdout_raw"] += len(eval_rows)
        source_profile["quality_buckets"][quality_bucket] += len(source_rows)

    split_rows: dict[str, list[dict[str, Any]]] = {}
    policy_summary: dict[str, Any] = {}
    for split_name, split_input_rows in raw_split_rows.items():
        split_rows[split_name], policy_summary[split_name] = _select_rows_by_quality_policy(
            split_input_rows,
            keep_rates=policies[split_name],
        )

    for split_name, split_rows_list in split_rows.items():
        for row in split_rows_list:
            source = str(row.get("source", "")).strip() or "__missing_source__"
            per_source[source][split_name] += 1

    finalized_per_source: dict[str, Any] = {}
    for source in sorted(per_source):
        profile = per_source[source]
        finalized_per_source[source] = {
            "total": profile["total"],
            "train_raw": profile["train_raw"],
            "dev_raw": profile["dev_raw"],
            "eval_holdout_raw": profile["eval_holdout_raw"],
            "train": profile["train"],
            "dev": profile["dev"],
            "eval_holdout": profile["eval_holdout"],
            "quality_buckets": dict(profile["quality_buckets"]),
        }

    summary = {
        "total_rows": len(rows),
        "train_rows": len(split_rows["train"]),
        "dev_rows": len(split_rows["dev"]),
        "eval_holdout_rows": len(split_rows["eval_holdout"]),
        "raw_train_rows": len(raw_split_rows["train"]),
        "raw_dev_rows": len(raw_split_rows["dev"]),
        "raw_eval_holdout_rows": len(raw_split_rows["eval_holdout"]),
        "ratios": {
            "train": train_ratio,
            "dev": dev_ratio,
            "eval_holdout": eval_ratio,
        },
        "seed": seed,
        "quality_policy": policies,
        "quality_policy_summary": policy_summary,
        "quality_bucket_totals": dict(Counter(_quality_bucket(row) for row in rows)),
        "per_source": finalized_per_source,
    }
    return split_rows, summary


def build_holdout_split(
    *,
    sft_path: Path,
    out_dir: Path,
    train_ratio: float,
    dev_ratio: float,
    eval_ratio: float,
    seed: int,
    quality_policies: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    rows = load_jsonl_rows(sft_path)
    split_rows, summary = split_sft_rows(
        rows,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        eval_ratio=eval_ratio,
        seed=seed,
        quality_policies=quality_policies,
    )

    train_path = out_dir / "sft_train.jsonl"
    dev_path = out_dir / "sft_dev.jsonl"
    eval_path = out_dir / "sft_eval_holdout.jsonl"
    write_jsonl_rows(train_path, split_rows["train"])
    write_jsonl_rows(dev_path, split_rows["dev"])
    write_jsonl_rows(eval_path, split_rows["eval_holdout"])

    summary.update(
        {
            "sft_path": str(sft_path),
            "out_dir": str(out_dir),
            "train_path": str(train_path),
            "dev_path": str(dev_path),
            "eval_holdout_path": str(eval_path),
        }
    )
    (out_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
