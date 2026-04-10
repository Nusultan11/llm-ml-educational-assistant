import json
from pathlib import Path

import typer

from llm_ml_assistant.utils.artifacts import load_serving_manifest, manifest_path_for


def main(
    artifacts_dir: Path = typer.Option(Path("artifacts"), "--artifacts-dir"),
):
    index_path = artifacts_dir / "rag_index.faiss"
    chunks_path = artifacts_dir / "rag_chunks.json"
    manifest_path = manifest_path_for(artifacts_dir)

    missing = [str(path) for path in [index_path, chunks_path, manifest_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Serving bundle is incomplete:\n- " + "\n- ".join(missing))

    manifest = load_serving_manifest(manifest_path)
    summary = {
        "artifacts_dir": str(artifacts_dir),
        "index_exists": index_path.exists(),
        "chunks_exists": chunks_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "chunk_count": manifest.get("artifacts", {}).get("chunk_count", 0),
        "generator_loading": manifest.get("online_pipeline", {}).get("generator_loading", ""),
        "allowed_steps": manifest.get("online_pipeline", {}).get("allowed_steps", []),
        "forbidden_steps": manifest.get("online_pipeline", {}).get("forbidden_steps", []),
    }

    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    typer.echo("\nServing bundle is valid.")


if __name__ == "__main__":
    typer.run(main)
