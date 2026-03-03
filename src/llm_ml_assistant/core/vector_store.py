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

    def _normalize(self, vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms