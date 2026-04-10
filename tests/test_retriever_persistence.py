import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.data.ingestion import DocumentRecord


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
            DocumentRecord(
                doc_id="rag_intro_txt",
                source_path="docs/rag_intro.txt",
                source_name="rag_intro.txt",
                title="What is RAG",
                text="RAG combines retrieval with generation.",
                section="What is RAG",
            ),
            DocumentRecord(
                doc_id="transformers_txt",
                source_path="docs/transformers.txt",
                source_name="transformers.txt",
                title="Transformers",
                text="Transformer models process tokens with attention.",
                section="Transformers",
            ),
            DocumentRecord(
                doc_id="misc_txt",
                source_path="docs/misc.txt",
                source_name="misc.txt",
                title="Misc",
                text="This sentence is unrelated to the topic.",
                section="Misc",
            ),
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

            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)
            self.assertIsInstance(payload[0], dict)
            self.assertEqual(payload[0]["doc_id"], "rag_intro_txt")
            self.assertEqual(payload[0]["source_name"], "rag_intro.txt")
            self.assertIn("chunk_id", payload[0])
            self.assertIn("start_char", payload[0])
            self.assertIn("end_char", payload[0])
            self.assertIn("prev_chunk_id", payload[0])
            self.assertIn("next_chunk_id", payload[0])

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

    def test_index_documents_accepts_document_records(self):
        config = SimpleNamespace(
            rag=SimpleNamespace(
                chunk_size=12,
                chunk_overlap=2,
                top_k=2,
                retrieval_mode="vector",
                rrf_k=60,
            ),
            embeddings=SimpleNamespace(name="fake"),
        )

        docs = [
            DocumentRecord(
                doc_id="rag_intro_txt",
                source_path="notes/rag_intro.txt",
                source_name="rag_intro.txt",
                title="What is RAG",
                text="RAG combines retrieval with generation for better answers.",
                section="What is RAG",
            )
        ]

        retriever = Retriever(config, embedding_model=FakeEmbeddingModel())
        retriever.index_documents(docs)

        self.assertGreaterEqual(len(retriever.chunk_records), 1)
        first = retriever.chunk_records[0]
        self.assertEqual(first.doc_id, "rag_intro_txt")
        self.assertEqual(first.source_path, "notes/rag_intro.txt")
        self.assertEqual(first.source_name, "rag_intro.txt")
        self.assertEqual(first.title, "What is RAG")
        self.assertIsNotNone(first.chunk_id)
        self.assertEqual(first.section, "What is RAG")

        retrieved = retriever.retrieve_records("What is RAG?")
        self.assertGreaterEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].doc_id, "rag_intro_txt")

    def test_load_supports_legacy_string_chunks_payload(self):
        config = SimpleNamespace(
            rag=SimpleNamespace(
                chunk_size=500,
                chunk_overlap=0,
                top_k=1,
                retrieval_mode="vector",
                rrf_k=60,
            ),
            embeddings=SimpleNamespace(name="fake"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            index_path = tmp_path / "rag_index.faiss"
            chunks_path = tmp_path / "rag_chunks.json"

            retriever = Retriever(config, embedding_model=FakeEmbeddingModel())
            retriever.index_documents(["RAG combines retrieval with generation."])
            retriever.save(index_path=index_path, chunks_path=chunks_path)

            chunks_path.write_text(
                json.dumps(["legacy chunk one", "legacy chunk two"]),
                encoding="utf-8",
            )

            restored = Retriever(config, embedding_model=FakeEmbeddingModel())
            restored.load(index_path=index_path, chunks_path=chunks_path)

            self.assertEqual(restored.text_chunks, ["legacy chunk one", "legacy chunk two"])
            self.assertEqual(restored.chunk_records[0].doc_id, "legacy_document_0000")
            self.assertEqual(restored.chunk_records[1].chunk_id, "legacy_document_0001__0000")

    def test_reranker_reorders_candidates_after_first_retrieval(self):
        config = SimpleNamespace(
            rag=SimpleNamespace(
                chunk_size=500,
                chunk_overlap=0,
                top_k=1,
                retrieval_mode="vector",
                rrf_k=60,
                reranker_enabled=True,
                reranker_type="token_overlap",
                reranker_candidate_k=3,
            ),
            embeddings=SimpleNamespace(name="fake"),
        )

        docs = [
            "This chunk discusses embeddings and model setup in general terms.",
            "RAG reduces hallucinations because answers are grounded in retrieved context.",
            "This chunk explains indexing and FAISS usage.",
        ]

        retriever = Retriever(config, embedding_model=ConstantEmbeddingModel())
        retriever.index_documents(docs)

        results = retriever.retrieve("Why does RAG reduce hallucinations?")

        self.assertEqual(len(results), 1)
        self.assertIn("reduces hallucinations", results[0])

    def test_quality_gate_marks_weak_retrieval_as_insufficient(self):
        config = SimpleNamespace(
            rag=SimpleNamespace(
                chunk_size=500,
                chunk_overlap=0,
                top_k=2,
                retrieval_mode="vector",
                rrf_k=60,
                reranker_enabled=False,
                reranker_type="token_overlap",
                reranker_candidate_k=None,
                quality_gate_enabled=True,
                quality_gate_min_score=0.35,
                quality_gate_min_coverage=0.3,
                quality_gate_min_strong_results=1,
            ),
            embeddings=SimpleNamespace(name="fake"),
        )

        retriever = Retriever(config, embedding_model=ConstantEmbeddingModel())
        retriever.index_documents(
            [
                "This chunk is about transformers and tokenization.",
                "Another chunk about embeddings and vector search.",
            ]
        )

        contexts, quality = retriever.retrieve_with_diagnostics(
            "What are the limits of QLoRA for long-context fine-tuning?"
        )

        self.assertEqual(len(contexts), 2)
        self.assertFalse(quality.sufficient)
        self.assertIn("partially overlaps", quality.reason)

    def test_quality_gate_allows_strong_retrieval(self):
        config = SimpleNamespace(
            rag=SimpleNamespace(
                chunk_size=500,
                chunk_overlap=0,
                top_k=1,
                retrieval_mode="vector",
                rrf_k=60,
                reranker_enabled=True,
                reranker_type="token_overlap",
                reranker_candidate_k=3,
                quality_gate_enabled=True,
                quality_gate_min_score=0.35,
                quality_gate_min_coverage=0.2,
                quality_gate_min_strong_results=1,
            ),
            embeddings=SimpleNamespace(name="fake"),
        )

        retriever = Retriever(config, embedding_model=ConstantEmbeddingModel())
        retriever.index_documents(
            [
                "RAG reduces hallucinations because answers are grounded in retrieved context.",
                "This chunk explains chunk overlap and indexing.",
            ]
        )

        contexts, quality = retriever.retrieve_with_diagnostics(
            "Why does RAG reduce hallucinations?"
        )

        self.assertEqual(len(contexts), 1)
        self.assertTrue(quality.sufficient)
        self.assertIn("passed", quality.reason)


if __name__ == "__main__":
    unittest.main()
