import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in cmd)


def _run_step(name: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n=== {name} ===")
    print(_format_cmd(cmd))
    started = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - started

    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")

    print(f"{name} completed in {elapsed:.1f}s")


def _ensure_paths_exist(paths: list[Path], label: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{label}: missing expected files:\n- " + "\n- ".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local end-to-end pipeline: prepare -> clean -> index -> archive."
    )

    parser.add_argument("--config", default="configs/colab_light.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--processed-clean-dir", default="data/processed_v2_clean")
    parser.add_argument("--rag-docs-dir", default="data/rag_docs_v2_clean")

    parser.add_argument("--max-openassistant", type=int, default=30000)
    parser.add_argument("--max-dolly", type=int, default=15000)
    parser.add_argument("--max-stackoverflow", type=int, default=30000)
    parser.add_argument("--max-arxiv", type=int, default=30000)
    parser.add_argument("--no-openassistant", action="store_true")
    parser.add_argument("--no-dolly", action="store_true")
    parser.add_argument("--stackoverflow-path", default="data/raw/stackoverflow.jsonl")
    parser.add_argument("--arxiv-path", default="data/raw/arxiv.jsonl")

    parser.add_argument("--min-rag-chars", type=int, default=80)
    parser.add_argument("--min-instruction-chars", type=int, default=10)
    parser.add_argument("--min-response-chars", type=int, default=40)
    parser.add_argument("--max-non-ascii-ratio", type=float, default=0.30)
    parser.add_argument("--noise-pattern", default=r"(?:http[s]?://|<[^>]+>|\bN/A\b|\blorem ipsum\b)")

    parser.add_argument("--snapshot-label", default="local_pipeline")
    parser.add_argument("--snapshot-notes", default="automated local pipeline run")

    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable

    env = os.environ.copy()
    src_path = str(root / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"

    processed_dir = root / args.processed_dir
    processed_clean_dir = root / args.processed_clean_dir
    rag_docs_dir = root / args.rag_docs_dir
    artifacts_dir = root / args.artifacts_dir
    config_path = root / args.config

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if not args.skip_prepare:
        prepare_cmd = [
            python_exec,
            str(root / "scripts" / "prepare_datasets.py"),
            "--out-dir",
            args.processed_dir,
            "--max-openassistant",
            str(args.max_openassistant),
            "--max-dolly",
            str(args.max_dolly),
            "--max-stackoverflow",
            str(args.max_stackoverflow),
            "--max-arxiv",
            str(args.max_arxiv),
            "--stackoverflow-path",
            args.stackoverflow_path,
            "--arxiv-path",
            args.arxiv_path,
        ]
        if args.no_openassistant:
            prepare_cmd.append("--no-openassistant")
        if args.no_dolly:
            prepare_cmd.append("--no-dolly")

        _run_step("Prepare datasets", prepare_cmd, env)
        _ensure_paths_exist(
            [
                processed_dir / "rag_corpus.jsonl",
                processed_dir / "sft_instructions.jsonl",
                processed_dir / "summary.json",
            ],
            "Prepare datasets",
        )

    if not args.skip_clean:
        clean_cmd = [
            python_exec,
            str(root / "scripts" / "clean_processed_datasets.py"),
            "--in-dir",
            args.processed_dir,
            "--out-dir",
            args.processed_clean_dir,
            "--rag-docs-dir",
            args.rag_docs_dir,
            "--min-rag-chars",
            str(args.min_rag_chars),
            "--min-instruction-chars",
            str(args.min_instruction_chars),
            "--min-response-chars",
            str(args.min_response_chars),
            "--max-non-ascii-ratio",
            str(args.max_non_ascii_ratio),
            "--noise-pattern",
            args.noise_pattern,
        ]
        _run_step("Clean datasets", clean_cmd, env)
        _ensure_paths_exist(
            [
                processed_clean_dir / "rag_corpus.jsonl",
                processed_clean_dir / "sft_instructions.jsonl",
                processed_clean_dir / "cleaning_summary.json",
            ],
            "Clean datasets",
        )

    if not args.skip_index:
        Path(args.artifacts_dir).mkdir(parents=True, exist_ok=True)

        index_cmd = [
            python_exec,
            "-m",
            "llm_ml_assistant.cli",
            "index",
            "--config",
            args.config,
            "--data-dir",
            args.rag_docs_dir,
            "--artifacts-dir",
            args.artifacts_dir,
            "--rebuild",
        ]
        _run_step("Build index", index_cmd, env)
        _ensure_paths_exist(
            [
                artifacts_dir / "rag_index.faiss",
                artifacts_dir / "rag_chunks.json",
                artifacts_dir / "rag_manifest.json",
            ],
            "Build index",
        )

    if not args.skip_archive:
        archive_cmd = [
            python_exec,
            str(root / "scripts" / "archive_artifacts.py"),
            "--artifacts-dir",
            args.artifacts_dir,
            "--config",
            args.config,
            "--data-dir",
            args.rag_docs_dir,
            "--label",
            args.snapshot_label,
            "--notes",
            args.snapshot_notes,
        ]
        _run_step("Archive artifacts", archive_cmd, env)
        _ensure_paths_exist([artifacts_dir / "latest_snapshot.json"], "Archive artifacts")

    print("\nPipeline finished successfully.")
    print(f"Processed: {processed_dir}")
    print(f"Cleaned:   {processed_clean_dir}")
    print(f"RAG docs:  {rag_docs_dir}")
    print(f"Artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
