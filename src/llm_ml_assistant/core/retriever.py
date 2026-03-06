import json
from pathlib import Path

from llm_ml_assistant.core.keyword_index import KeywordIndex
from llm_ml_assistant.core.vector_store import VectorStore
from llm_ml_assistant.data.chunking import chunk_text
from llm_ml_assistant.models.embeddings import EmbeddingModel


class Retriever:
    def __init__(self, config, embedding_model=None):
        self.chunk_size = config.rag.chunk_size
        self.chunk_overlap = config.rag.chunk_overlap
        self.top_k = config.rag.top_k
        self.retrieval_mode = getattr(config.rag, "retrieval_mode", "vector")
        self.rrf_k = getattr(config.rag, "rrf_k", 60)

        if self.retrieval_mode not in {"vector", "hybrid"}:
            raise ValueError("retrieval_mode must be 'vector' or 'hybrid'")

        self.embedding_model = embedding_model or EmbeddingModel(config.embeddings.name)
        self.text_chunks = []
        self.vector_store = None
        self.keyword_index = None

    def index_documents(self, texts: list[str]):
        all_chunks = []

        for text in texts:
            chunks = chunk_text(
                text,
                self.chunk_size,
                self.chunk_overlap,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks were created from input documents.")

        self.text_chunks = all_chunks
        vectors = self.embedding_model.encode_documents(all_chunks)

        self.vector_store = VectorStore(vectors.shape[1])
        self.vector_store.add(vectors)

        if self.retrieval_mode == "hybrid":
            self.keyword_index = KeywordIndex()
            self.keyword_index.build(self.text_chunks)

    def retrieve(self, query: str):
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
            json.dumps(self.text_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, index_path: Path, chunks_path: Path):
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError("Index files were not found. Run index command first.")

        self.vector_store = VectorStore.load(index_path)
        self.text_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        if self.retrieval_mode == "hybrid":
            self.keyword_index = KeywordIndex()
            self.keyword_index.build(self.text_chunks)

    def _retrieve_vector_only(self, query: str) -> list[str]:
        query_vector = self.embedding_model.encode_queries([query])
        _scores, indices = self.vector_store.search(query_vector, self.top_k)

        valid_chunks = []
        for idx in indices[0]:
            if idx == -1:
                continue
            valid_chunks.append(self.text_chunks[idx])

        return valid_chunks

    def _retrieve_hybrid(self, query: str) -> list[str]:
        candidate_k = max(self.top_k * 3, self.top_k)

        query_vector = self.embedding_model.encode_queries([query])
        _scores, vector_indices = self.vector_store.search(query_vector, candidate_k)
        vector_ranked = [idx for idx in vector_indices[0] if idx != -1]

        keyword_ranked = self.keyword_index.search(query, candidate_k) if self.keyword_index else []

        fused_indices = self._fuse_rankings(vector_ranked, keyword_ranked)
        top_indices = fused_indices[: self.top_k]

        return [self.text_chunks[idx] for idx in top_indices]

    def _fuse_rankings(self, vector_ranked: list[int], keyword_ranked: list[int]) -> list[int]:
        scores = {}

        for rank, idx in enumerate(vector_ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, idx in enumerate(keyword_ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in ranked]
