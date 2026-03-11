import argparse
import copy
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_ml_assistant.utils.ablation import (  # noqa: E402
    generate_retrieval_variants,
    parse_csv_ints,
    parse_csv_strings,
    safe_run_label,
)


def _resolve_from_root(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def _as_cli_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_step(name: str, cmd: list[str], env: dict[str, str]) -> tuple[int, float]:
    print(f"\n=== {name} ===")
    print(_format_cmd(cmd))

    started = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - started

    status = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
    print(f"{name} {status} in {elapsed:.1f}s")

    return result.returncode, elapsed


def _read_latest_snapshot_name(artifacts_dir: Path) -> str:
    latest_path = artifacts_dir / "latest_snapshot.json"
    if not latest_path.exists():
        return ""

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    snapshot_dir = payload.get("snapshot_dir", "")
    if not snapshot_dir:
        return ""

    return Path(snapshot_dir).name


def _write_variant_config(base_config: dict, out_path: Path, mode: str, chunk_size: int, overlap: int, top_k: int) -> None:
    updated = copy.deepcopy(base_config)
    rag = updated.setdefault("rag", {})
    rag["retrieval_mode"] = mode
    rag["chunk_size"] = chunk_size
    rag["chunk_overlap"] = overlap
    rag["top_k"] = top_k

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(updated, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _write_leaderboard_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "tag",
        "retrieval_mode",
        "chunk_size",
        "chunk_overlap",
        "top_k",
        "queries",
        "hit_rate",
        "mrr",
        "snapshot",
        "pipeline_profile_reused",
        "pipeline_sec",
        "eval_sec",
        "total_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_leaderboard_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retrieval Ablation Leaderboard",
        "",
        "| Rank | Tag | Mode | Chunk | Overlap | TopK | HitRate | MRR | Snapshot |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {rank} | `{tag}` | `{retrieval_mode}` | {chunk_size} | {chunk_overlap} | {top_k} | "
            "{hit_rate:.3f} | {mrr:.3f} | `{snapshot}` |".format(
                rank=row["rank"],
                tag=row["tag"],
                retrieval_mode=row["retrieval_mode"],
                chunk_size=row["chunk_size"],
                chunk_overlap=row["chunk_overlap"],
                top_k=row["top_k"],
                hit_rate=row["hit_rate"],
                mrr=row["mrr"],
                snapshot=row.get("snapshot", ""),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile_key(mode: str, chunk_size: int, chunk_overlap: int) -> tuple[str, int, int]:
    return mode, chunk_size, chunk_overlap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval ablation over chunk_size/chunk_overlap/top_k variants and log metrics."
    )

    parser.add_argument("--base-config", default="configs/colab_light.yaml")
    parser.add_argument("--processed-clean-dir", default="data/processed_v2_clean_plus")
    parser.add_argument("--rag-docs-dir", default="data/rag_docs_v2_clean_plus")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--eval", default="data/processed_v2_clean/eval_auto_qa.json")

    parser.add_argument("--out-dir", default="reports/retrieval_metrics/ablation")
    parser.add_argument("--history-path", default="reports/retrieval_metrics/history.jsonl")

    parser.add_argument("--tag-prefix", default="ablation")
    parser.add_argument("--snapshot-notes", default="automated retrieval ablation run")

    parser.add_argument("--chunk-sizes", default="420,520,700")
    parser.add_argument("--chunk-overlaps", default="40,80,120")
    parser.add_argument("--top-ks", default="3,5,8")
    parser.add_argument("--retrieval-modes", default="hybrid")

    parser.add_argument("--run-clean", action="store_true", help="Run cleaning step on each index profile.")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of variants (0 means all).")
    parser.add_argument("--dry-run", action="store_true", help="Only generate configs/plan, do not execute pipeline.")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_config_path = _resolve_from_root(args.base_config)
    processed_clean_dir = _resolve_from_root(args.processed_clean_dir)
    rag_docs_dir = _resolve_from_root(args.rag_docs_dir)
    artifacts_dir = _resolve_from_root(args.artifacts_dir)
    eval_path = _resolve_from_root(args.eval)
    out_dir = _resolve_from_root(args.out_dir)
    history_path = _resolve_from_root(args.history_path)

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not eval_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Eval file not found: {eval_path}")
    if not args.run_clean and not rag_docs_dir.exists() and not args.dry_run:
        raise FileNotFoundError(f"RAG docs directory not found: {rag_docs_dir}")

    chunk_sizes = parse_csv_ints(args.chunk_sizes, "chunk_sizes", min_value=1)
    chunk_overlaps = parse_csv_ints(args.chunk_overlaps, "chunk_overlaps", min_value=0)
    top_ks = parse_csv_ints(args.top_ks, "top_ks", min_value=1)
    retrieval_modes = parse_csv_strings(args.retrieval_modes, "retrieval_modes")

    variants = generate_retrieval_variants(
        chunk_sizes=chunk_sizes,
        chunk_overlaps=chunk_overlaps,
        top_ks=top_ks,
        retrieval_modes=retrieval_modes,
    )

    if args.max_runs > 0:
        variants = variants[: args.max_runs]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / run_id
    configs_dir = run_dir / "configs"
    metrics_dir = run_dir / "metrics"

    configs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}{os.pathsep}{existing_pythonpath}"

    python_exec = sys.executable
    results: list[dict] = []

    unique_profiles = {
        _profile_key(v.retrieval_mode, v.chunk_size, v.chunk_overlap)
        for v in variants
    }
    profile_state: dict[tuple[str, int, int], dict] = {}

    print(f"Run id: {run_id}")
    print(f"Profiles: {len(unique_profiles)} | Variants: {len(variants)}")

    for index, variant in enumerate(variants, start=1):
        tag = safe_run_label(
            f"{args.tag_prefix}_{variant.retrieval_mode}_c{variant.chunk_size}_o{variant.chunk_overlap}_k{variant.top_k}",
            max_len=64,
        )
        config_path = configs_dir / f"{index:02d}_{tag}.yaml"
        metrics_path = metrics_dir / f"{index:02d}_{tag}.json"

        _write_variant_config(
            base_config=base_config,
            out_path=config_path,
            mode=variant.retrieval_mode,
            chunk_size=variant.chunk_size,
            overlap=variant.chunk_overlap,
            top_k=variant.top_k,
        )

        key = _profile_key(variant.retrieval_mode, variant.chunk_size, variant.chunk_overlap)
        state = profile_state.get(key)
        profile_reused = state is not None

        item = {
            "variant_index": index,
            "tag": tag,
            "retrieval_mode": variant.retrieval_mode,
            "chunk_size": variant.chunk_size,
            "chunk_overlap": variant.chunk_overlap,
            "top_k": variant.top_k,
            "config": _as_cli_path(config_path),
            "metrics_file": _as_cli_path(metrics_path),
            "pipeline_profile_reused": profile_reused,
            "status": "dry_run" if args.dry_run else "pending",
        }

        if args.dry_run:
            results.append(item)
            continue

        if state is None:
            snapshot_label = safe_run_label(f"{run_id}_{index:02d}_{tag}", max_len=80)
            snapshot_notes = f"{args.snapshot_notes}; profile={key}; seed_tag={tag}"

            pipeline_cmd = [
                python_exec,
                str(ROOT / "scripts" / "run_local_pipeline.py"),
                "--config",
                _as_cli_path(config_path),
                "--artifacts-dir",
                _as_cli_path(artifacts_dir),
                "--processed-clean-dir",
                _as_cli_path(processed_clean_dir),
                "--rag-docs-dir",
                _as_cli_path(rag_docs_dir),
                "--skip-prepare",
                "--snapshot-label",
                snapshot_label,
                "--snapshot-notes",
                snapshot_notes,
            ]
            if not args.run_clean:
                pipeline_cmd.append("--skip-clean")

            pipeline_code, pipeline_sec = _run_step(
                f"Build profile {len(profile_state)+1}/{len(unique_profiles)} [{key}]",
                pipeline_cmd,
                env,
            )
            snapshot_name = _read_latest_snapshot_name(artifacts_dir) if pipeline_code == 0 else ""

            state = {
                "status": "ok" if pipeline_code == 0 else "pipeline_failed",
                "pipeline_sec": round(pipeline_sec, 3),
                "snapshot": snapshot_name,
            }
            profile_state[key] = state

        item["snapshot"] = state.get("snapshot", "")
        item["pipeline_sec"] = 0.0 if profile_reused else state.get("pipeline_sec", 0.0)

        if state.get("status") != "ok":
            item["status"] = "pipeline_failed"
            results.append(item)
            continue

        eval_cmd = [
            python_exec,
            str(ROOT / "scripts" / "evaluate_artifacts_retrieval.py"),
            "--config",
            _as_cli_path(config_path),
            "--artifacts-dir",
            _as_cli_path(artifacts_dir),
            "--eval",
            _as_cli_path(eval_path),
            "--tag",
            tag,
            "--snapshot-label",
            state.get("snapshot", ""),
            "--out",
            _as_cli_path(metrics_path),
            "--history-path",
            _as_cli_path(history_path),
        ]

        eval_code, eval_sec = _run_step(f"Evaluate variant {index}/{len(variants)} [{tag}]", eval_cmd, env)
        item["eval_sec"] = round(eval_sec, 3)
        item["total_sec"] = round(item["pipeline_sec"] + eval_sec, 3)

        if eval_code != 0 or not metrics_path.exists():
            item["status"] = "eval_failed"
            results.append(item)
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        item["queries"] = metrics.get("queries", 0)
        item["hit_rate"] = metrics.get("hit_rate", 0.0)
        item["mrr"] = metrics.get("mrr", 0.0)
        item["status"] = "ok"
        results.append(item)

    successful = [r for r in results if r.get("status") == "ok"]
    leaderboard = sorted(successful, key=lambda r: (r.get("mrr", 0.0), r.get("hit_rate", 0.0)), reverse=True)

    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "base_config": _as_cli_path(base_config_path),
        "artifacts_dir": _as_cli_path(artifacts_dir),
        "eval_file": _as_cli_path(eval_path),
        "history_path": _as_cli_path(history_path),
        "dry_run": args.dry_run,
        "profiles_total": len(unique_profiles),
        "profiles_built": len([s for s in profile_state.values() if s.get("status") == "ok"]),
        "variants_total": len(variants),
        "variants_successful": len(successful),
        "results": results,
    }

    summary_path = run_dir / "run_summary.json"
    leaderboard_csv = run_dir / "leaderboard.csv"
    leaderboard_md = run_dir / "leaderboard.md"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_leaderboard_csv(leaderboard_csv, leaderboard)
    _write_leaderboard_md(leaderboard_md, leaderboard)

    if leaderboard:
        best = leaderboard[0]
        print("\nBest variant:")
        print(
            f"- {best['tag']} | mode={best['retrieval_mode']} | "
            f"chunk={best['chunk_size']} overlap={best['chunk_overlap']} top_k={best['top_k']} | "
            f"HitRate={best['hit_rate']:.3f} MRR={best['mrr']:.3f}"
        )

    print("\nAblation run completed.")
    print(f"Summary: {summary_path}")
    print(f"Leaderboard CSV: {leaderboard_csv}")
    print(f"Leaderboard MD: {leaderboard_md}")


if __name__ == "__main__":
    main()
