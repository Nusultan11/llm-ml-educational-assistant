from pathlib import Path

from llm_ml_assistant.utils.config import load_config
from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.core.rag_pipeline import RAGPipeline


def main():
    config = load_config(Path("configs/base.yaml"))

    print("=== CONFIG LOADED ===")
    print("Project:", config.project.name)
    print("Embedding model:", config.embeddings.name)
    print("Top-K:", config.rag.top_k)
    print()

    docs = [
        """
        Retrieval Augmented Generation (RAG) is a technique that combines
        information retrieval with text generation.
        It first retrieves relevant documents from a knowledge base,
        then uses a language model to generate an answer grounded in that context.
        """,
        """
        Neural networks are computational models inspired by the human brain.
        They are widely used in deep learning for computer vision,
        natural language processing, and speech recognition.
        """,
        """
        Transformers are neural network architectures based on self-attention.
        They are dominant in NLP tasks such as translation,
        summarization, and question answering.
        """,
    ]

    retriever = Retriever(config)
    prompt_builder = PromptBuilder()
    rag = RAGPipeline(retriever=retriever, prompt_builder=prompt_builder)

    print("Indexing documents...")
    rag.index(docs)
    print("Index built successfully.\n")

    query = "Explain what RAG is."
    out = rag.build(query)

    print("=== TOP-K CONTEXT ===")
    for i, c in enumerate(out["contexts"], 1):
        print(f"\n[{i}]\n{c.strip()}")

    print("\n=== BUILT PROMPT (preview) ===")
    # Чтобы не засорять консоль, показываем первые ~800 символов
    preview = out["prompt"][:800]
    print(preview)
    if len(out["prompt"]) > 800:
        print("\n... (truncated)")


if __name__ == "__main__":
    main()