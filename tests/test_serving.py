import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from llm_ml_assistant.core.context_assembler import ContextAssembler
from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.core.serving import OnlineRAGService
from llm_ml_assistant.data.ingestion import DocumentRecord
from llm_ml_assistant.utils.artifacts import (
    build_serving_manifest,
    manifest_path_for,
    save_serving_manifest,
)


class ConstantEmbeddingModel:
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 1.0, 1.0] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 1.0, 1.0] for _ in texts], dtype=np.float32)


class FakeGenerator:
    def __init__(self, answer_text: str, calls: list[str]):
        self.answer_text = answer_text
        self.calls = calls

    def generate(self, prompt: str, max_tokens: int = 512):
        self.calls.append(prompt)
        return self.answer_text


class OnlineServingTests(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            rag=SimpleNamespace(
                chunk_size=700,
                chunk_overlap=40,
                top_k=2,
                retrieval_mode="hybrid",
                rrf_k=60,
                reranker_enabled=True,
                reranker_type="token_overlap",
                reranker_candidate_k=4,
                quality_gate_enabled=True,
                quality_gate_min_score=0.35,
                quality_gate_min_coverage=0.2,
                quality_gate_min_strong_results=1,
                context_max_blocks=3,
                context_max_chars=1800,
                context_max_chunks_per_doc=2,
                context_dedup_threshold=0.8,
                context_expand_neighbors=True,
            ),
            embeddings=SimpleNamespace(name="fake"),
            model=SimpleNamespace(name="fake-generator", device="cpu", max_tokens=64),
        )

    def _build_service(self, docs: list[DocumentRecord], answer_text: str = "[ASSISTANT]\nGrounded answer"):
        config = self._config()
        calls: list[str] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_dir = Path(tmp_dir)
            retriever = Retriever(config, embedding_model=ConstantEmbeddingModel())
            retriever.index_documents(docs)
            index_path = artifacts_dir / "rag_index.faiss"
            chunks_path = artifacts_dir / "rag_chunks.json"
            retriever.save(index_path=index_path, chunks_path=chunks_path)
            save_serving_manifest(
                manifest_path_for(artifacts_dir),
                build_serving_manifest(
                    config=config,
                    artifacts_dir=artifacts_dir,
                    index_path=index_path,
                    chunks_path=chunks_path,
                    chunk_count=len(retriever.chunk_records),
                    data_dir=Path("data/examples"),
                    config_path=Path("configs/colab_light.yaml"),
                ),
            )

            service = OnlineRAGService.from_artifacts(
                config=config,
                artifacts_dir=artifacts_dir,
                prompt_builder=PromptBuilder(),
                context_assembler=ContextAssembler(
                    max_blocks=3,
                    max_chars=1800,
                    max_chunks_per_doc=2,
                    dedup_threshold=0.8,
                    expand_neighbors=True,
                    chunk_size_hint=config.rag.chunk_size,
                ),
                retriever=Retriever(config, embedding_model=ConstantEmbeddingModel()),
                generator_factory=lambda: FakeGenerator(answer_text, calls),
            )
            return service, calls

    def test_retrieval_only_online_path_does_not_load_generator(self):
        docs = [
            DocumentRecord(
                doc_id="rag_intro",
                source_path="docs/rag_intro.txt",
                source_name="rag_intro.txt",
                title="What is RAG",
                text="RAG reduces hallucinations because answers are grounded in retrieved context.",
                section="What is RAG",
            )
        ]

        service, calls = self._build_service(docs)
        result = service.answer("Why does RAG reduce hallucinations?", mode="retrieval_only")

        self.assertEqual(result.mode, "retrieval_only")
        self.assertTrue(result.attribution["grounded"])
        self.assertGreaterEqual(result.attribution["evidence_count"], 1)
        self.assertEqual(calls, [])

    def test_rag_online_path_uses_lazy_generator(self):
        docs = [
            DocumentRecord(
                doc_id="rag_intro",
                source_path="docs/rag_intro.txt",
                source_name="rag_intro.txt",
                title="What is RAG",
                text="RAG reduces hallucinations because answers are grounded in retrieved context.",
                section="What is RAG",
            )
        ]

        service, calls = self._build_service(docs)
        result = service.answer("Why does RAG reduce hallucinations?", mode="rag")

        self.assertEqual(result.mode, "rag")
        self.assertEqual(result.answer, "Grounded answer")
        self.assertEqual(len(calls), 1)
        self.assertTrue(service.serving_summary()["manifest_available"])
        self.assertEqual(service.serving_summary()["pipeline_role"], "online")


if __name__ == "__main__":
    unittest.main()
