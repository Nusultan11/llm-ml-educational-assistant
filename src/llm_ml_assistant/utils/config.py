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


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)