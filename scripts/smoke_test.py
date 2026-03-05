from dataclasses import dataclass
from typing import List

from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.core.rag_pipeline import RAGPipeline


@dataclass
class MockRetriever:
    top_k: int = 2

    def __post_init__(self):
        self.docs: List[str] = []

    def index_documents(self, docs: List[str]):
        self.docs = docs

    def retrieve(self, query: str) -> List[str]:
        query_terms = set(query.lower().split())
        scored = []

        for doc in self.docs:
            doc_terms = set(doc.lower().split())
            score = len(query_terms.intersection(doc_terms))
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored if score > 0][: self.top_k]


class MockGenerator:
    def generate(self, prompt: str) -> str:
        if "[CONTEXT" in prompt:
            return "Smoke test passed: answer generated from retrieved context."
        return "Smoke test failed: no context in prompt."


def main():
    docs = [
        "RAG combines retrieval and generation.",
        "Transformers are used in modern NLP.",
        "Vector search helps find relevant context.",
    ]

    retriever = MockRetriever(top_k=2)
    prompt_builder = PromptBuilder()
    generator = MockGenerator()

    rag = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        generator=generator,
    )

    rag.index(docs)
    answer = rag.ask("What is RAG?")

    print(answer)


if __name__ == "__main__":
    main()
