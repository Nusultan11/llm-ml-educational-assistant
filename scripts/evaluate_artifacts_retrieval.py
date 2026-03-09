import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer

from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.utils.config import load_config


def _load_eval(eval_path: Path) -> list[dict]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Eval file must be a non-empty list")
    return data


def _rank_of_hit(contexts: list[str], expected_substring: str) -> int | None:
    expected = expected_substring.lower()
    for i, ctx in enumerate(contexts, start=1):
        if expected in ctx.lower():
            return i
    return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main(
    config_path: Path = typer.Option(Path("configs/colab_light.yaml"), "--config"),
    artifacts_dir: Path = typer.Option(Path("artifacts"), "--artifacts-dir"),
    eval_path: Path = typer.Option(Path("data/processed_v2_clean/eval_auto_qa.json"), "--eval"),
    out_path: Path | None = typer.Option(None, "--out", help="Optional JSON file to save metrics."),
    history_path: Path | None = typer.Option(
        None,
        "--history-path",
        help="Optional JSONL file to append one metrics record per run.",
    ),
    snapshot_label: str = typer.Option("", "--snapshot-label", help="Snapshot id/label for traceability."),
    tag: str = typer.Option("", "--tag", help="Short run tag, e.g. baseline/v2_clean_plus."),
):
    config = load_config(config_path)
    index_path = artifacts_dir / "rag_index.faiss"
    chunks_path = artifacts_dir / "rag_chunks.json"

    retriever = Retriever(config)
    retriever.load(index_path=index_path, chunks_path=chunks_path)

    eval_items = _load_eval(eval_path)

    hits = 0
    reciprocal_rank_sum = 0.0

    for item in eval_items:
        query = item["query"]
        expected_substring = item["expected_substring"]

        contexts = retriever.retrieve(query)
        rank = _rank_of_hit(contexts, expected_substring)

        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / rank

    total = len(eval_items)
    hit_rate = hits / total
    mrr = reciprocal_rank_sum / total

    result = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tag": tag,
        "snapshot_label": snapshot_label,
        "config": str(config_path),
        "artifacts_dir": str(artifacts_dir),
        "eval_file": str(eval_path),
        "embedding_model": config.embeddings.name,
        "retrieval_mode": getattr(config.rag, "retrieval_mode", "vector"),
        "top_k": config.rag.top_k,
        "queries": total,
        "hits": hits,
        "hit_rate": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "offline_flags": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", ""),
        },
    }

    print(f"Config: {config_path}")
    print(f"Artifacts dir: {artifacts_dir}")
    print(f"Eval file: {eval_path}")
    print(f"Queries: {total}")
    print(f"HitRate@{config.rag.top_k}: {hit_rate:.3f}")
    print(f"MRR@{config.rag.top_k}: {mrr:.3f}")

    if out_path is not None:
        _write_json(out_path, result)
        print(f"Saved metrics JSON: {out_path}")

    if history_path is not None:
        _append_jsonl(history_path, result)
        print(f"Appended metrics history: {history_path}")


if __name__ == "__main__":
    typer.run(main)