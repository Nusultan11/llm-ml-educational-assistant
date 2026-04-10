import logging
from pathlib import Path
from types import SimpleNamespace

import typer

from llm_ml_assistant.core.context_assembler import AssembledContextBlock, ContextAssembler
from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.core.serving import OnlineRAGService
from llm_ml_assistant.data.ingestion import DocumentRecord, build_document_record
from llm_ml_assistant.utils.artifacts import (
    build_serving_manifest,
    manifest_path_for,
    save_serving_manifest,
)
from llm_ml_assistant.utils.config import load_config

app = typer.Typer(help="CLI for llm-ml-assistant")


def _read_documents(data_dir: Path) -> list[DocumentRecord]:
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    files = sorted(
        [*data_dir.rglob("*.txt"), *data_dir.rglob("*.md")],
        key=lambda p: str(p),
    )
    docs = []

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(build_document_record(path=path, text=text, root_dir=data_dir))

    if not docs:
        raise ValueError(f"No .txt or .md documents found in {data_dir}.")

    return docs


def _index_paths(artifacts_dir: Path) -> tuple[Path, Path]:
    return artifacts_dir / "rag_index.faiss", artifacts_dir / "rag_chunks.json"


def _build_logger(logs_dir: Path) -> tuple[logging.Logger, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "cli.log"
    logger = logging.getLogger("llm_ml_assistant.cli")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger, log_path


def _print_contexts(contexts: list[str]):
    typer.echo("\n=== Retrieved Contexts ===")
    for i, ctx in enumerate(contexts, start=1):
        typer.echo(f"\n[{i}]\n{ctx.strip()}")


def _print_source_attribution(blocks: list[AssembledContextBlock], quality) -> None:
    typer.echo("\n=== Source Attribution ===")
    typer.echo(f"grounded: {'yes' if quality.sufficient else 'no'}")
    typer.echo(f"evidence_count: {len(blocks)}")
    typer.echo(f"retrieval_reason: {quality.reason}")

    if not blocks:
        return

    typer.echo("\nSources:")
    for block in blocks:
        source = block.source_name or block.doc_id
        section = f" | section={block.section}" if block.section else ""
        chunk_ids = ", ".join(block.chunk_ids)
        typer.echo(
            f"- {source} | chunks={chunk_ids}{section} | chars={block.start_char}:{block.end_char}"
        )


def _build_context_assembler(config) -> ContextAssembler:
    return ContextAssembler(
        max_blocks=getattr(config.rag, "context_max_blocks", 3),
        max_chars=getattr(config.rag, "context_max_chars", 1800),
        max_chunks_per_doc=getattr(config.rag, "context_max_chunks_per_doc", 2),
        dedup_threshold=getattr(config.rag, "context_dedup_threshold", 0.8),
        expand_neighbors=getattr(config.rag, "context_expand_neighbors", True),
        chunk_size_hint=config.rag.chunk_size,
    )


@app.command()
def index(
    config_path: Path = typer.Option(Path("configs/base.yaml"), "--config"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    artifacts_dir: Path | None = typer.Option(None, "--artifacts-dir"),
    rebuild: bool = typer.Option(False, "--rebuild"),
):
    """Build and persist retrieval index from local documents."""
    config = load_config(config_path)
    logger, log_path = _build_logger(Path(config.paths.logs_dir))

    try:
        resolved_data_dir = data_dir or Path(config.paths.data_dir)
        resolved_artifacts_dir = artifacts_dir or Path(config.paths.artifacts_dir)
        index_path, chunks_path = _index_paths(resolved_artifacts_dir)

        if index_path.exists() and chunks_path.exists() and not rebuild:
            typer.echo(
                f"Index already exists at {index_path}. Use --rebuild to recreate it."
            )
            raise typer.Exit(code=0)

        docs = _read_documents(resolved_data_dir)

        retriever = Retriever(config)
        retriever.index_documents(docs)
        retriever.save(index_path=index_path, chunks_path=chunks_path)
        manifest_path = manifest_path_for(resolved_artifacts_dir)
        save_serving_manifest(
            manifest_path,
            build_serving_manifest(
                config=config,
                artifacts_dir=resolved_artifacts_dir,
                index_path=index_path,
                chunks_path=chunks_path,
                chunk_count=len(retriever.chunk_records),
                data_dir=resolved_data_dir,
                config_path=config_path,
            ),
        )

        logger.info("Indexed %s documents from %s", len(docs), resolved_data_dir)
        typer.echo(f"Indexed {len(docs)} documents.")
        typer.echo(f"Saved index to: {index_path}")
        typer.echo(f"Saved chunks to: {chunks_path}")
        typer.echo(f"Saved manifest to: {manifest_path}")
    except typer.Exit:
        raise
    except Exception as exc:
        logger.exception("Index command failed")
        typer.echo(f"Index failed: {exc}", err=True)
        typer.echo(f"See logs: {log_path}", err=True)
        raise typer.Exit(code=1)


@app.command()
def ask(
    query: str = typer.Argument(..., help="User question"),
    config_path: Path = typer.Option(Path("configs/base.yaml"), "--config"),
    artifacts_dir: Path | None = typer.Option(None, "--artifacts-dir"),
    mode: str = typer.Option("rag", "--mode", help="rag or retrieval_only"),
    show_contexts: bool = typer.Option(False, "--show-contexts"),
):
    """Load persisted index and answer a question."""
    config = load_config(config_path)
    logger, log_path = _build_logger(Path(config.paths.logs_dir))

    try:
        if mode not in {"rag", "retrieval_only"}:
            raise ValueError("mode must be 'rag' or 'retrieval_only'")

        resolved_artifacts_dir = artifacts_dir or Path(config.paths.artifacts_dir)
        index_path, chunks_path = _index_paths(resolved_artifacts_dir)

        assembler = _build_context_assembler(config)
        service = OnlineRAGService.from_artifacts(
            config=config,
            artifacts_dir=resolved_artifacts_dir,
            prompt_builder=PromptBuilder(),
            context_assembler=assembler,
        )
        result = service.answer(query=query, mode=mode, show_contexts=show_contexts)
        blocks = [
            AssembledContextBlock(
                rank=source["rank"],
                doc_id=source["doc_id"],
                source_path=source["source_path"],
                source_name=source["source_name"],
                title=source["title"],
                section=source["section"],
                chunk_ids=source["chunk_ids"],
                start_char=source["start_char"],
                end_char=source["end_char"],
                text=source["text"],
                rendered_text="",
            )
            for source in result.attribution.get("sources", [])
        ]
        contexts = result.contexts
        quality = SimpleNamespace(**result.retrieval_quality)

        if show_contexts:
            _print_contexts(contexts)

        if result.mode == "retrieval_only" and mode == "retrieval_only":
            logger.info("Answered query in retrieval_only mode: %s", query)
            if not contexts:
                typer.echo("No relevant contexts found.")
                _print_source_attribution(blocks, quality)
                return
            typer.echo("\n=== Retrieval-Only Answer ===")
            typer.echo(result.answer)
            _print_source_attribution(blocks, quality)
            return

        if result.mode == "retrieval_only" and mode == "rag":
            logger.info("Blocked rag generation due to weak retrieval: %s", quality.reason)
            typer.echo(result.answer)
            if not show_contexts:
                _print_contexts(contexts)
            _print_source_attribution(blocks, quality)
            return

        logger.info("Answered query in rag mode: %s", query)
        typer.echo(result.answer)
        _print_source_attribution(blocks, quality)

    except Exception as exc:
        logger.exception("Ask command failed")
        typer.echo(f"Ask failed: {exc}", err=True)
        typer.echo(f"See logs: {log_path}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
