from pathlib import Path
import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    name: str
    version: str


class ModelConfig(BaseModel):
    name: str
    device: str
    max_tokens: int


class RAGConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_mode: str = "vector"
    rrf_k: int = 60
    reranker_enabled: bool = False
    reranker_type: str = "token_overlap"
    reranker_candidate_k: int | None = None
    quality_gate_enabled: bool = False
    quality_gate_min_score: float = 0.2
    quality_gate_min_coverage: float = 0.2
    quality_gate_min_strong_results: int = 1
    context_max_blocks: int = 3
    context_max_chars: int = 1800
    context_max_chunks_per_doc: int = 2
    context_dedup_threshold: float = 0.8
    context_expand_neighbors: bool = True


class EmbeddingConfig(BaseModel):
    name: str


class PathsConfig(BaseModel):
    data_dir: str
    artifacts_dir: str
    logs_dir: str


class Config(BaseModel):
    project: ProjectConfig
    model: ModelConfig
    rag: RAGConfig
    embeddings: EmbeddingConfig
    paths: PathsConfig


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
