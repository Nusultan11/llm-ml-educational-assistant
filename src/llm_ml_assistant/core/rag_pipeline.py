from dataclasses import dataclass
from typing import List

from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.models.generator import Generator


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        generator: Generator,
    ):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator

    def index(self, docs: List[str]):
        self.retriever.index_documents(docs)

    def ask(self, query: str) -> str:
        contexts = self.retriever.retrieve(query)
        prompt = self.prompt_builder.build(query, contexts)
        answer = self.generator.generate(prompt)
        return answer