import json
from pathlib import Path

from llm_ml_assistant.core.vector_store import VectorStore
from llm_ml_assistant.data.chunking import chunk_text
from llm_ml_assistant.models.embeddings import EmbeddingModel


class Retriever:
    def __init__(self, config, embedding_model=None):
        self.chunk_size = config.rag.chunk_size
        self.chunk_overlap = config.rag.chunk_overlap
        self.top_k = config.rag.top_k

        self.embedding_model = embedding_model or EmbeddingModel(config.embeddings.name)
        self.text_chunks = []
        self.vector_store = None

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
        vectors = self.embedding_model.encode(all_chunks)

        self.vector_store = VectorStore(vectors.shape[1])
        self.vector_store.add(vectors)

    def retrieve(self, query: str):
        if self.vector_store is None:
            raise RuntimeError("Retriever is not indexed. Call index_documents() first.")

        query_vector = self.embedding_model.encode([query])
        scores, indices = self.vector_store.search(query_vector, self.top_k)

        valid_chunks = []

        for idx in indices[0]:
            if idx == -1:
                continue
            valid_chunks.append(self.text_chunks[idx])

        return valid_chunks

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
