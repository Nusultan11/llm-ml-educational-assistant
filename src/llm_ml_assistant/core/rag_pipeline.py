from dataclasses import dataclass
from typing import List

from llm_ml_assistant.core.context_assembler import ContextAssembler
from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.models.generator import Generator


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        generator: Generator,
        context_assembler: ContextAssembler | None = None,
    ):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.context_assembler = context_assembler

    def index(self, docs: List[str]):
        self.retriever.index_documents(docs)

    def ask(self, query: str) -> str:
        if self.context_assembler is None or not hasattr(self.retriever, "retrieve_records"):
            contexts = self.retriever.retrieve(query)
        else:
            records = self.retriever.retrieve_records(query)
            contexts = self.context_assembler.assemble(
                records,
                getattr(self.retriever, "chunk_records", records),
            )
        prompt = self.prompt_builder.build(query, contexts)
        answer = self.generator.generate(prompt)
        return answer
