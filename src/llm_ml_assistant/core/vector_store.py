from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray):
        vectors = self._normalize(vectors)
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int):
        query_vector = self._normalize(query_vector)
        scores, indices = self.index.search(query_vector, top_k)
        return scores, indices

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        index = faiss.read_index(str(path))
        store = cls(index.d)
        store.index = index
        return store

    def _normalize(self, vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms
