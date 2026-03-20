from pathlib import Path
import sys
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_ml_assistant.core.prompt_builder import PromptBuilder
from llm_ml_assistant.core.retriever import Retriever
from llm_ml_assistant.models.generator import Generator
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


def _extract_assistant_answer(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    # If model echoes full prompt, keep only the tail after the last assistant tag.
    if "[ASSISTANT]" in raw:
        raw = raw.rsplit("[ASSISTANT]", 1)[1].strip()

    # Drop any leaked template tags.
    raw = re.split(r"\[(?:SYSTEM|CONTEXT|USER|ASSISTANT)\]", raw, maxsplit=1)[0].strip()
    return raw


config_override = os.getenv("LLM_CONFIG_PATH", "").strip()
if config_override:
    config_path = Path(config_override)
else:
    config_path = ROOT_DIR / "configs" / "colab_light.yaml"

config = load_config(config_path)
retriever = Retriever(config)
prompt_builder = PromptBuilder()
generator: Generator | None = None

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

retriever.load(index_path=index_path, chunks_path=chunks_path)

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
    return {"ok": True}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    global generator

    query = request.query.strip()
    if not query:
        return {"error": "query must not be empty"}

    if request.mode not in {"retrieval_only", "rag"}:
        return {"error": "mode must be 'retrieval_only' or 'rag'"}

    contexts = retriever.retrieve(query)

    if request.mode == "retrieval_only":
        return {
            "mode": request.mode,
            "answer": "Use retrieved contexts as evidence for the final response.",
            "contexts": contexts if request.show_contexts else [],
        }

    try:
        if generator is None:
            generator = Generator(
                model_name=config.model.name,
                device=config.model.device,
            )

        prompt = prompt_builder.build(query, contexts)
        raw_answer = generator.generate(prompt, max_tokens=config.model.max_tokens)
        answer = _extract_assistant_answer(raw_answer)
    except Exception as exc:
        return {
            "mode": "retrieval_only",
            "answer": (
                "RAG generation failed on this machine. "
                "Returning retrieval-only contexts. "
                f"Error: {exc}"
            ),
            "contexts": contexts if request.show_contexts else [],
            "rag_error": str(exc),
        }

    return {
        "mode": request.mode,
        "answer": answer,
        "contexts": contexts if request.show_contexts else [],
    }


@app.post("/api/chat")
def api_chat(request: ChatRequest) -> dict:
    result = ask(
        AskRequest(
            query=request.message,
            mode=request.mode,
            show_contexts=False,
        )
    )
    return {"reply": result.get("answer", ""), "mode": result.get("mode", request.mode)}
