import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from llm_ml_assistant.utils.config import load_config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label.strip())
    return cleaned[:80] or "snapshot"


def main():
    parser = argparse.ArgumentParser(
        description="Archive current RAG artifacts into a timestamped snapshot with metadata."
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory with rag_index.faiss and rag_chunks.json")
    parser.add_argument("--config", default="configs/base.yaml", help="Config used to build index")
    parser.add_argument("--data-dir", default=None, help="Data directory used for indexing (optional override)")
    parser.add_argument("--label", default="manual", help="Human-readable label for this snapshot")
    parser.add_argument("--notes", default="", help="Optional free-form notes")

    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    index_path = artifacts_dir / "rag_index.faiss"
    chunks_path = artifacts_dir / "rag_chunks.json"
    manifest_path = artifacts_dir / "rag_manifest.json"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts. Expected both {index_path} and {chunks_path}."
        )

    config = load_config(args.config)
    resolved_data_dir = str(Path(args.data_dir)) if args.data_dir else config.paths.data_dir

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = safe_label(args.label)
    snapshot_dir = artifacts_dir / "snapshots" / f"{timestamp}__{label}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    snap_index = snapshot_dir / "rag_index.faiss"
    snap_chunks = snapshot_dir / "rag_chunks.json"
    snap_manifest = snapshot_dir / "rag_manifest.json"

    shutil.copy2(index_path, snap_index)
    shutil.copy2(chunks_path, snap_chunks)
    if manifest_path.exists():
        shutil.copy2(manifest_path, snap_manifest)

    chunks = json.loads(snap_chunks.read_text(encoding="utf-8"))

    metadata = {
        "created_at_utc": timestamp,
        "label": label,
        "notes": args.notes,
        "source": {
            "artifacts_dir": str(artifacts_dir),
            "config": args.config,
            "data_dir": resolved_data_dir,
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
        },
        "artifacts": {
            "rag_index": {
                "path": str(snap_index),
                "size_bytes": snap_index.stat().st_size,
                "sha256": sha256_file(snap_index),
            },
            "rag_chunks": {
                "path": str(snap_chunks),
                "size_bytes": snap_chunks.stat().st_size,
                "sha256": sha256_file(snap_chunks),
                "chunk_count": len(chunks),
            },
        },
    }

    if snap_manifest.exists():
        metadata["artifacts"]["rag_manifest"] = {
            "path": str(snap_manifest),
            "size_bytes": snap_manifest.stat().st_size,
            "sha256": sha256_file(snap_manifest),
        }

    metadata_path = snapshot_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_path = artifacts_dir / "latest_snapshot.json"
    latest_path.write_text(json.dumps({"snapshot_dir": str(snapshot_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Snapshot created: {snapshot_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
