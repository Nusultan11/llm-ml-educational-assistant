from pathlib import Path
import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.core.context_assembler import ContextAssembler
from llm_ml_assistant.core.serving import OnlineRAGService
from llm_ml_assistant.utils.config import load_config


class AskRequest(BaseModel):
    query: str
    mode: str = "retrieval_only"
    show_contexts: bool = True


class ChatRequest(BaseModel):
    message: str
    mode: str = "rag"


def _index_paths(artifacts_dir: Path) -> tuple[Path, Path]:
    return artifacts_dir / "rag_index.faiss", artifacts_dir / "rag_chunks.json"


def _build_context_assembler(config) -> ContextAssembler:
    return ContextAssembler(
        max_blocks=getattr(config.rag, "context_max_blocks", 3),
        max_chars=getattr(config.rag, "context_max_chars", 1800),
        max_chunks_per_doc=getattr(config.rag, "context_max_chunks_per_doc", 2),
        dedup_threshold=getattr(config.rag, "context_dedup_threshold", 0.8),
        expand_neighbors=getattr(config.rag, "context_expand_neighbors", True),
        chunk_size_hint=config.rag.chunk_size,
    )


config_override = os.getenv("LLM_CONFIG_PATH", "").strip()
if config_override:
    config_path = Path(config_override)
else:
    config_path = ROOT_DIR / "configs" / "colab_light.yaml"

config = load_config(config_path)
prompt_builder = PromptBuilder()
context_assembler = _build_context_assembler(config)

artifacts_override = os.getenv("LLM_ARTIFACTS_DIR", "").strip()
if artifacts_override:
    artifacts_dir = Path(artifacts_override)
else:
    artifacts_dir = ROOT_DIR / config.paths.artifacts_dir

index_path, chunks_path = _index_paths(artifacts_dir)

if not index_path.exists() or not chunks_path.exists():
    raise FileNotFoundError(
        "RAG artifacts not found. Set LLM_ARTIFACTS_DIR to an existing artifacts folder "
        f"(index: {index_path}, chunks: {chunks_path})."
    )

service = OnlineRAGService.from_artifacts(
    config=config,
    artifacts_dir=artifacts_dir,
    prompt_builder=prompt_builder,
    context_assembler=context_assembler,
)

app = FastAPI(title="llm-ml-assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "serving": service.serving_summary()}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    query = request.query.strip()
    if not query:
        return {"error": "query must not be empty"}

    if request.mode not in {"retrieval_only", "rag"}:
        return {"error": "mode must be 'retrieval_only' or 'rag'"}

    result = service.answer(
        query=query,
        mode=request.mode,
        show_contexts=request.show_contexts,
    )
    return result.to_dict()


@app.post("/api/chat")
def api_chat(request: ChatRequest) -> dict:
    result = ask(
        AskRequest(
            query=request.message,
            mode=request.mode,
            show_contexts=False,
        )
    )
    attribution = result.get("attribution", {})
    return {
        "reply": result.get("answer", ""),
        "mode": result.get("mode", request.mode),
        "sources": result.get("sources", []),
        "serving": service.serving_summary(),
        "grounding": {
            "grounded": attribution.get("grounded", False),
            "evidence_count": attribution.get("evidence_count", 0),
            "reason": attribution.get("reason", ""),
        },
        "retrieval_quality": result.get("retrieval_quality", {}),
    }
