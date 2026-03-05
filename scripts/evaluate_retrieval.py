import json
from pathlib import Path

import typer

from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.utils.config import load_config


def _load_documents(data_dir: Path) -> list[str]:
    files = sorted([*data_dir.rglob("*.txt"), *data_dir.rglob("*.md")], key=lambda p: str(p))
    docs = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(text)
    if not docs:
        raise ValueError(f"No .txt or .md documents found in {data_dir}")
    return docs


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


def main(
    config_path: Path = typer.Option(Path("configs/base.yaml"), "--config"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    eval_path: Path = typer.Option(Path("data/examples/eval_qa.json"), "--eval"),
):
    config = load_config(config_path)
    resolved_data_dir = data_dir or Path(config.paths.data_dir)

    docs = _load_documents(resolved_data_dir)
    eval_items = _load_eval(eval_path)

    retriever = Retriever(config)
    retriever.index_documents(docs)

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

    print(f"Config: {config_path}")
    print(f"Data dir: {resolved_data_dir}")
    print(f"Eval file: {eval_path}")
    print(f"Queries: {total}")
    print(f"HitRate@{config.rag.top_k}: {hit_rate:.3f}")
    print(f"MRR@{config.rag.top_k}: {mrr:.3f}")


if __name__ == "__main__":
    typer.run(main)
