from pathlib import Path

import typer

from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.core.rag_pipeline import RAGPipeline
from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.models.generator import Generator
from llm_ml_assistant.utils.config import load_config

app = typer.Typer(help="CLI for llm-ml-assistant")


def _read_documents(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        raise typer.BadParameter(f"Data directory does not exist: {data_dir}")

    files = sorted(
        [*data_dir.rglob("*.txt"), *data_dir.rglob("*.md")],
        key=lambda p: str(p),
    )
    docs = []

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(text)

    if not docs:
        raise typer.BadParameter(
            f"No .txt or .md documents found in {data_dir}."
        )

    return docs


def _index_paths(artifacts_dir: Path) -> tuple[Path, Path]:
    return artifacts_dir / "rag_index.faiss", artifacts_dir / "rag_chunks.json"


@app.command()
def index(
    config_path: Path = typer.Option(Path("configs/base.yaml"), "--config"),
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    artifacts_dir: Path | None = typer.Option(None, "--artifacts-dir"),
    rebuild: bool = typer.Option(False, "--rebuild"),
):
    """Build and persist retrieval index from local documents."""
    config = load_config(config_path)
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

    typer.echo(f"Indexed {len(docs)} documents.")
    typer.echo(f"Saved index to: {index_path}")
    typer.echo(f"Saved chunks to: {chunks_path}")


@app.command()
def ask(
    query: str = typer.Argument(..., help="User question"),
    config_path: Path = typer.Option(Path("configs/base.yaml"), "--config"),
    artifacts_dir: Path | None = typer.Option(None, "--artifacts-dir"),
):
    """Load persisted index and answer a question."""
    config = load_config(config_path)
    resolved_artifacts_dir = artifacts_dir or Path(config.paths.artifacts_dir)
    index_path, chunks_path = _index_paths(resolved_artifacts_dir)

    retriever = Retriever(config)
    retriever.load(index_path=index_path, chunks_path=chunks_path)

    prompt_builder = PromptBuilder()
    generator = Generator(
        model_name=config.model.name,
        device=config.model.device,
    )
    rag = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        generator=generator,
    )

    answer = rag.ask(query)
    typer.echo(answer)


if __name__ == "__main__":
    app()
