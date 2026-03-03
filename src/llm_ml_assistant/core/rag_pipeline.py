from dataclasses import dataclass
from typing import List, Dict, Any

from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.core.prompt_builder import PromptBuilder


@dataclass
class RAGPipeline:
    retriever: Retriever
    prompt_builder: PromptBuilder

    def index(self, docs: List[str]) -> None:
        self.retriever.index_documents(docs)

    def build(self, query: str) -> Dict[str, Any]:
        contexts = self.retriever.retrieve(query)
        prompt = self.prompt_builder.build(query, contexts)
        return {
            "query": query,
            "contexts": contexts,
            "prompt": prompt,
        }