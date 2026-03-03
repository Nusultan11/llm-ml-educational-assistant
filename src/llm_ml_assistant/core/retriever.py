from llm_ml_assistant.data.chunking import chunk_text
from llm_ml_assistant.models.embeddings import EmbeddingModel
from llm_ml_assistant.core.vector_store import VectorStore


class Retriever:
    def __init__(self, config):
        self.chunk_size = config.rag.chunk_size
        self.chunk_overlap = config.rag.chunk_overlap
        self.top_k = config.rag.top_k

        self.embedding_model = EmbeddingModel(config.embeddings.name)
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

        self.text_chunks = all_chunks
        vectors = self.embedding_model.encode(all_chunks)

        self.vector_store = VectorStore(vectors.shape[1])
        self.vector_store.add(vectors)

    def retrieve(self, query: str):
        query_vector = self.embedding_model.encode([query])
        scores, indices = self.vector_store.search(query_vector, self.top_k)

        valid_chunks = []

        for idx in indices[0]:
            if idx == -1:
                continue
            valid_chunks.append(self.text_chunks[idx])

        return valid_chunks