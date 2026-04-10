from dataclasses import dataclass
from pathlib import Path

from llm_ml_assistant.core.context_assembler import AssembledContextBlock, ContextAssembler
from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.models.generator import Generator
from llm_ml_assistant.utils.artifacts import load_serving_manifest, manifest_path_for


@dataclass(frozen=True)
class OnlineAnswer:
    mode: str
    answer: str
    contexts: list[str]
    retrieval_quality: dict
    attribution: dict
    rag_error: str = ""

    def to_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "answer": self.answer,
            "contexts": self.contexts,
            "retrieval_quality": self.retrieval_quality,
            "attribution": self.attribution,
            "sources": self.attribution.get("sources", []),
        }
        if self.rag_error:
            payload["rag_error"] = self.rag_error
        return payload


class OnlineRAGService:
    def __init__(
        self,
        *,
        config,
        artifacts_dir: Path,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        context_assembler: ContextAssembler,
        generator_factory=None,
    ):
        self.config = config
        self.artifacts_dir = artifacts_dir
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.context_assembler = context_assembler
        self.generator_factory = generator_factory or (
            lambda: Generator(
                model_name=self.config.model.name,
                device=self.config.model.device,
            )
        )
        self._generator = None
        self.manifest_path = manifest_path_for(artifacts_dir)
        self.manifest_available = self.manifest_path.exists()
        self.manifest = (
            load_serving_manifest(self.manifest_path)
            if self.manifest_available
            else self._build_legacy_manifest()
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        config,
        artifacts_dir: Path,
        prompt_builder: PromptBuilder,
        context_assembler: ContextAssembler,
        retriever: Retriever | None = None,
        generator_factory=None,
    ):
        retriever = retriever or Retriever(config)
        index_path = artifacts_dir / "rag_index.faiss"
        chunks_path = artifacts_dir / "rag_chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "Online serving requires prebuilt artifacts. "
                f"Missing index or chunks under {artifacts_dir}."
            )
        retriever.load(index_path=index_path, chunks_path=chunks_path)
        return cls(
            config=config,
            artifacts_dir=artifacts_dir,
            retriever=retriever,
            prompt_builder=prompt_builder,
            context_assembler=context_assembler,
            generator_factory=generator_factory,
        )

    def answer(self, query: str, mode: str = "retrieval_only", show_contexts: bool = True) -> OnlineAnswer:
        if mode not in {"retrieval_only", "rag"}:
            raise ValueError("mode must be 'retrieval_only' or 'rag'")

        records = self.retriever.retrieve_records(query)
        quality = self.retriever.assess_retrieval_quality(query, records)
        blocks = self.context_assembler.assemble_blocks(records, self.retriever.chunk_records)
        contexts = [block.rendered_text for block in blocks]
        attribution = self._build_attribution(blocks, quality)

        if mode == "retrieval_only":
            return OnlineAnswer(
                mode=mode,
                answer="Use retrieved contexts as evidence for the final response.",
                contexts=contexts if show_contexts else [],
                retrieval_quality=quality.to_dict(),
                attribution=attribution,
            )

        if not quality.sufficient:
            return OnlineAnswer(
                mode="retrieval_only",
                answer=(
                    "I found only partially relevant context, so I cannot answer reliably yet. "
                    f"{quality.reason}"
                ),
                contexts=contexts if show_contexts else [],
                retrieval_quality=quality.to_dict(),
                attribution=attribution,
            )

        try:
            prompt = self.prompt_builder.build(query, contexts)
            raw_answer = self.generator.generate(prompt, max_tokens=self.config.model.max_tokens)
            answer = self._extract_assistant_answer(raw_answer)
            return OnlineAnswer(
                mode="rag",
                answer=answer,
                contexts=contexts if show_contexts else [],
                retrieval_quality=quality.to_dict(),
                attribution=attribution,
            )
        except Exception as exc:
            return OnlineAnswer(
                mode="retrieval_only",
                answer=(
                    "RAG generation failed on this machine. "
                    "Returning retrieval-only contexts. "
                    f"Error: {exc}"
                ),
                contexts=contexts if show_contexts else [],
                retrieval_quality=quality.to_dict(),
                attribution=attribution,
                rag_error=str(exc),
            )

    @property
    def generator(self):
        if self._generator is None:
            self._generator = self.generator_factory()
        return self._generator

    def serving_summary(self) -> dict:
        return {
            "pipeline_role": "online",
            "artifacts_dir": str(self.artifacts_dir),
            "manifest_available": self.manifest_available,
            "generator_loading": self.manifest.get("online_pipeline", {}).get(
                "generator_loading",
                "lazy_on_first_rag_request",
            ),
            "allowed_steps": self.manifest.get("online_pipeline", {}).get("allowed_steps", []),
            "forbidden_steps": self.manifest.get("online_pipeline", {}).get("forbidden_steps", []),
        }

    def _build_attribution(self, blocks: list[AssembledContextBlock], quality) -> dict:
        return {
            "grounded": quality.sufficient,
            "evidence_count": len(blocks),
            "reason": quality.reason,
            "sources": [block.to_source_dict() for block in blocks],
        }

    def _build_legacy_manifest(self) -> dict:
        return {
            "schema_version": "legacy",
            "online_pipeline": {
                "allowed_steps": [
                    "load_artifacts",
                    "retrieve",
                    "rerank",
                    "quality_gate",
                    "context_assembly",
                    "generate",
                    "source_attribution",
                ],
                "forbidden_steps": [
                    "prepare_datasets",
                    "clean_processed_datasets",
                    "reindex",
                    "reembed",
                    "ablation",
                    "evaluation",
                ],
                "generator_loading": "lazy_on_first_rag_request",
            },
        }

    def _extract_assistant_answer(self, text: str) -> str:
        import re

        raw = (text or "").strip()
        if not raw:
            return ""
        if "[ASSISTANT]" in raw:
            raw = raw.rsplit("[ASSISTANT]", 1)[1].strip()
        raw = re.split(r"\[(?:SYSTEM|CONTEXT|USER|ASSISTANT)\]", raw, maxsplit=1)[0].strip()
        return raw
