from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        prepared = self._prepare_documents(texts)
        return self.model.encode(prepared, convert_to_numpy=True)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prepared = self._prepare_queries(texts)
        return self.model.encode(prepared, convert_to_numpy=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Backward-compatible alias used in tests and old call sites.
        return self.encode_documents(texts)

    def _prepare_documents(self, texts: list[str]) -> list[str]:
        if self._is_e5():
            return [f"passage: {t}" for t in texts]
        return texts

    def _prepare_queries(self, texts: list[str]) -> list[str]:
        if self._is_e5():
            return [f"query: {t}" for t in texts]
        return texts

    def _is_e5(self) -> bool:
        return "e5" in self.model_name.lower()
