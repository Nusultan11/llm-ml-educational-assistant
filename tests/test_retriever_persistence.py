import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from llm_ml_assistant.core.retriever import Retriever


class FakeEmbeddingModel:
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float(len(lower)),
                    float(lower.count("rag")),
                    float(lower.count("transformer")),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class ConstantEmbeddingModel:
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 1.0, 1.0] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 1.0, 1.0] for _ in texts], dtype=np.float32)


class RetrieverPersistenceTests(unittest.TestCase):
    def test_save_load_and_retrieve(self):
        config = SimpleNamespace(
            rag=SimpleNamespace(
                chunk_size=500,
                chunk_overlap=0,
                top_k=2,
                retrieval_mode="vector",
                rrf_k=60,
            ),
            embeddings=SimpleNamespace(name="fake"),
        )

        docs = [
            "RAG combines retrieval with generation.",
            "Transformer models process tokens with attention.",
            "This sentence is unrelated to the topic.",
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            index_path = tmp_path / "rag_index.faiss"
            chunks_path = tmp_path / "rag_chunks.json"

            retriever = Retriever(config, embedding_model=FakeEmbeddingModel())
            retriever.index_documents(docs)
            retriever.save(index_path=index_path, chunks_path=chunks_path)

            restored = Retriever(config, embedding_model=FakeEmbeddingModel())
            restored.load(index_path=index_path, chunks_path=chunks_path)

            results = restored.retrieve("What is RAG?")

            self.assertTrue(index_path.exists())
            self.assertTrue(chunks_path.exists())
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("RAG", results[0])

    def test_hybrid_mode_uses_keyword_signal(self):
        config = SimpleNamespace(
            rag=SimpleNamespace(
                chunk_size=500,
                chunk_overlap=0,
                top_k=1,
                retrieval_mode="hybrid",
                rrf_k=60,
            ),
            embeddings=SimpleNamespace(name="fake"),
        )

        docs = [
            "This paragraph is generic and does not mention the target.",
            "Another generic paragraph without special term.",
            "Hybrid retrieval should find foobar_token evidence quickly.",
        ]

        retriever = Retriever(config, embedding_model=ConstantEmbeddingModel())
        retriever.index_documents(docs)

        results = retriever.retrieve("Where is foobar_token mentioned?")

        self.assertEqual(len(results), 1)
        self.assertIn("foobar_token", results[0])


if __name__ == "__main__":
    unittest.main()
