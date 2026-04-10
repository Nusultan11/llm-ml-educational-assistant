import json
from pathlib import Path

from llm_ml_assistant.core.keyword_index import KeywordIndex
from llm_ml_assistant.core.retrieval_quality import RetrievalQuality, RetrievalQualityGate
from llm_ml_assistant.core.reranker import TokenOverlapReranker
from llm_ml_assistant.core.vector_store import VectorStore
from llm_ml_assistant.data.ingestion import (
    DocumentRecord,
    build_chunk_records,
    chunk_record_from_payload,
    infer_section,
    infer_title,
    legacy_text_to_chunk_record,
    make_doc_id,
)
from llm_ml_assistant.models.embeddings import EmbeddingModel


class Retriever:
    def __init__(self, config, embedding_model=None):
        self.chunk_size = config.rag.chunk_size
        self.chunk_overlap = config.rag.chunk_overlap
        self.top_k = config.rag.top_k
        self.retrieval_mode = getattr(config.rag, "retrieval_mode", "vector")
        self.rrf_k = getattr(config.rag, "rrf_k", 60)
        self.reranker_enabled = getattr(config.rag, "reranker_enabled", False)
        self.reranker_type = getattr(config.rag, "reranker_type", "token_overlap")
        self.reranker_candidate_k = getattr(config.rag, "reranker_candidate_k", None)
        self.quality_gate_enabled = getattr(config.rag, "quality_gate_enabled", False)
        self.quality_gate_min_score = getattr(config.rag, "quality_gate_min_score", 0.2)
        self.quality_gate_min_coverage = getattr(config.rag, "quality_gate_min_coverage", 0.2)
        self.quality_gate_min_strong_results = getattr(config.rag, "quality_gate_min_strong_results", 1)

        if self.retrieval_mode not in {"vector", "hybrid"}:
            raise ValueError("retrieval_mode must be 'vector' or 'hybrid'")
        if self.reranker_type not in {"token_overlap"}:
            raise ValueError("reranker_type must be 'token_overlap'")

        self.embedding_model = embedding_model or EmbeddingModel(config.embeddings.name)
        self.chunk_records = []
        self.text_chunks = []
        self.vector_store = None
        self.keyword_index = None
        self.reranker = TokenOverlapReranker() if self.reranker_enabled else None
        self.quality_gate = (
            RetrievalQualityGate(
                min_score=self.quality_gate_min_score,
                min_coverage=self.quality_gate_min_coverage,
                min_strong_results=self.quality_gate_min_strong_results,
            )
            if self.quality_gate_enabled
            else None
        )

    def index_documents(self, documents: list[str | DocumentRecord]):
        normalized_documents = self._normalize_documents(documents)
        all_records = []

        for document in normalized_documents:
            all_records.extend(
                build_chunk_records(
                    document,
                    self.chunk_size,
                    self.chunk_overlap,
                )
            )

        if not all_records:
            raise ValueError("No chunks were created from input documents.")

        self.chunk_records = all_records
        self.text_chunks = [record.text for record in all_records]
        vectors = self.embedding_model.encode_documents(self.text_chunks)

        self.vector_store = VectorStore(vectors.shape[1])
        self.vector_store.add(vectors)

        if self.retrieval_mode == "hybrid":
            self.keyword_index = KeywordIndex()
            self.keyword_index.build(self.text_chunks)

    def retrieve(self, query: str):
        return [record.text for record in self.retrieve_records(query)]

    def retrieve_records(self, query: str):
        if self.vector_store is None:
            raise RuntimeError("Retriever is not indexed. Call index_documents() first.")

        if self.retrieval_mode == "vector":
            return self._retrieve_vector_only(query)

        return self._retrieve_hybrid(query)

    def save(self, index_path: Path, chunks_path: Path):
        if self.vector_store is None:
            raise RuntimeError("Retriever is not indexed. Nothing to save.")

        self.vector_store.save(index_path)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.write_text(
            json.dumps(
                [record.to_dict() for record in self.chunk_records],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, index_path: Path, chunks_path: Path):
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError("Index files were not found. Run index command first.")

        self.vector_store = VectorStore.load(index_path)
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunk_records = self._load_chunk_records(payload)
        self.text_chunks = [record.text for record in self.chunk_records]

        if self.retrieval_mode == "hybrid":
            self.keyword_index = KeywordIndex()
            self.keyword_index.build(self.text_chunks)

    def _retrieve_vector_only(self, query: str):
        candidate_k = self._candidate_k(for_hybrid=False)
        query_vector = self.embedding_model.encode_queries([query])
        _scores, indices = self.vector_store.search(query_vector, candidate_k)

        valid_chunks = []
        for idx in indices[0]:
            if idx == -1:
                continue
            valid_chunks.append(self.chunk_records[idx])

        return self._apply_reranker(query, valid_chunks)

    def _retrieve_hybrid(self, query: str):
        candidate_k = self._candidate_k(for_hybrid=True)

        query_vector = self.embedding_model.encode_queries([query])
        _scores, vector_indices = self.vector_store.search(query_vector, candidate_k)
        vector_ranked = [idx for idx in vector_indices[0] if idx != -1]

        keyword_ranked = self.keyword_index.search(query, candidate_k) if self.keyword_index else []

        fused_indices = self._fuse_rankings(vector_ranked, keyword_ranked)
        candidate_records = [self.chunk_records[idx] for idx in fused_indices[:candidate_k]]

        return self._apply_reranker(query, candidate_records)

    def assess_retrieval_quality(self, query: str, chunk_records: list) -> RetrievalQuality:
        if self.quality_gate is None:
            return RetrievalQuality(
                sufficient=True,
                reason="Quality gate disabled.",
                best_score=1.0 if chunk_records else 0.0,
                average_score=1.0 if chunk_records else 0.0,
                best_coverage=1.0 if chunk_records else 0.0,
                strong_results=len(chunk_records),
                total_results=len(chunk_records),
            )
        return self.quality_gate.assess(query, chunk_records)

    def retrieve_with_diagnostics(self, query: str) -> tuple[list[str], RetrievalQuality]:
        records = self.retrieve_records(query)
        quality = self.assess_retrieval_quality(query, records)
        return [record.text for record in records], quality

    def _normalize_documents(self, documents: list[str | DocumentRecord]) -> list[DocumentRecord]:
        normalized = []

        for idx, document in enumerate(documents):
            if isinstance(document, DocumentRecord):
                normalized.append(document)
                continue

            text = str(document).strip()
            if not text:
                continue

            source_name = f"document_{idx:04d}.txt"
            normalized.append(
                DocumentRecord(
                    doc_id=make_doc_id(source_name, f"document_{idx:04d}"),
                    source_path=source_name,
                    source_name=source_name,
                    title=infer_title(text, fallback=f"document_{idx:04d}"),
                    text=text,
                    section=infer_section(text),
                )
            )

        return normalized

    def _load_chunk_records(self, payload: list) -> list:
        if not payload:
            return []

        first = payload[0]
        if isinstance(first, str):
            return [legacy_text_to_chunk_record(text, idx) for idx, text in enumerate(payload)]

        if isinstance(first, dict):
            return [chunk_record_from_payload(item) for item in payload]

        raise ValueError("Unsupported chunk payload format.")

    def _candidate_k(self, for_hybrid: bool) -> int:
        base_k = max(self.top_k * 3, self.top_k) if for_hybrid else self.top_k
        if self.reranker_enabled:
            configured = self.reranker_candidate_k or max(self.top_k * 4, base_k)
            return max(configured, base_k)
        return base_k

    def _apply_reranker(self, query: str, candidate_records: list):
        if not self.reranker_enabled or self.reranker is None:
            return candidate_records[: self.top_k]
        return self.reranker.rerank(query, candidate_records, self.top_k)

    def _fuse_rankings(self, vector_ranked: list[int], keyword_ranked: list[int]) -> list[int]:
        scores = {}

        for rank, idx in enumerate(vector_ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, idx in enumerate(keyword_ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in ranked]
