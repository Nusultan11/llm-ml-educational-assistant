import json
from datetime import datetime, timezone
from pathlib import Path


def manifest_path_for(artifacts_dir: Path) -> Path:
    return artifacts_dir / "rag_manifest.json"


def build_serving_manifest(
    *,
    config,
    artifacts_dir: Path,
    index_path: Path,
    chunks_path: Path,
    chunk_count: int,
    data_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts_dir": str(artifacts_dir),
        "artifacts": {
            "index_path": str(index_path),
            "chunks_path": str(chunks_path),
            "chunk_count": chunk_count,
        },
        "offline_pipeline": {
            "data_dir": str(data_dir) if data_dir else "",
            "config_path": str(config_path) if config_path else "",
            "steps": [
                "prepare_datasets",
                "clean_processed_datasets",
                "chunk_documents",
                "build_embeddings",
                "build_index",
                "archive_artifacts",
                "evaluate",
            ],
        },
        "online_pipeline": {
            "allowed_steps": [
                "load_artifacts",
                "retrieve",
                "rerank",
                "quality_gate",
                "context_assembly",
                "generate",
                "source_attribution",
            ],
            "forbidden_steps": [
                "prepare_datasets",
                "clean_processed_datasets",
                "reindex",
                "reembed",
                "ablation",
                "evaluation",
            ],
            "generator_loading": "lazy_on_first_rag_request",
        },
        "config": {
            "embedding_model": config.embeddings.name,
            "generator_model": config.model.name,
            "device": config.model.device,
            "max_tokens": config.model.max_tokens,
            "chunk_size": config.rag.chunk_size,
            "chunk_overlap": config.rag.chunk_overlap,
            "top_k": config.rag.top_k,
            "retrieval_mode": getattr(config.rag, "retrieval_mode", "vector"),
            "rrf_k": getattr(config.rag, "rrf_k", 60),
            "reranker_enabled": getattr(config.rag, "reranker_enabled", False),
            "quality_gate_enabled": getattr(config.rag, "quality_gate_enabled", False),
            "context_max_blocks": getattr(config.rag, "context_max_blocks", 0),
            "context_max_chars": getattr(config.rag, "context_max_chars", 0),
        },
    }


def save_serving_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_serving_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
