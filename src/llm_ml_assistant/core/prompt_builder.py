from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PromptBuilder:
    system_prompt: str = (
        "You are a helpful ML assistant. Answer using ONLY the provided context. "
        "If context is insufficient, say you don't know."
    )

    def build(self, query: str, contexts: List[str]) -> str:
        context_block = "\n\n".join(
            f"[CONTEXT {i+1}]\n{c.strip()}" for i, c in enumerate(contexts)
        )

        prompt = (
            f"[SYSTEM]\n{self.system_prompt}\n\n"
            f"[CONTEXT]\n{context_block}\n\n"
            f"[USER]\n{query}\n\n"
            f"[ASSISTANT]\n"
        )
        return prompt