# llm-ml-assistant

Minimal RAG assistant skeleton with local retrieval and LLM generation.

## What is implemented

- YAML + Pydantic config loading
- Document chunking and vector indexing (FAISS)
- Retrieval + prompt building
- Generator wrapper based on Hugging Face Transformers
- CLI commands for indexing and question answering

## Project layout

- `src/llm_ml_assistant/core` - retrieval, vector store, prompt, RAG pipeline
- `src/llm_ml_assistant/models` - embedding and generator models
- `src/llm_ml_assistant/utils` - config loading
- `src/llm_ml_assistant/cli.py` - CLI commands (`index`, `ask`)
- `configs/base.yaml` - base runtime config
- `scripts/smoke_test.py` - offline smoke test (no model downloads)
- `tests/` - unit tests for core logic

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

## Build persistent index

```bash
llm-ml-assistant index --config configs/base.yaml --data-dir data --artifacts-dir artifacts --rebuild
```

Notes:
- Reads `.txt` and `.md` files recursively from `data`.
- Saves `artifacts/rag_index.faiss` and `artifacts/rag_chunks.json`.

## Ask a question

```bash
llm-ml-assistant ask "What is RAG?" --config configs/base.yaml --artifacts-dir artifacts
```

Note: the first run may download models from Hugging Face, so internet access is required.

## Run legacy main demo

```bash
python -m llm_ml_assistant.main
```

## Run offline smoke test

```bash
python scripts/smoke_test.py
```

This test validates the pipeline flow (`index -> retrieve -> prompt -> answer`) using mock retriever and generator.

## Run unit tests

```bash
python -m unittest discover -s tests -v
```
