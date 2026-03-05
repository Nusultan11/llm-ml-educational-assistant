# llm-ml-assistant

Minimal RAG assistant skeleton with local retrieval and LLM generation.

## Features

- Config loading via YAML + Pydantic
- Document chunking and vector indexing (FAISS)
- Retrieval + prompt building
- Generator wrapper for Hugging Face Transformers
- CLI commands for indexing and question answering
- Persistent index save/load (`.faiss` + chunks JSON)
- Unit and integration tests

## Repository structure

- `src/llm_ml_assistant/core` - retrieval, vector store, prompt builder, RAG pipeline
- `src/llm_ml_assistant/models` - embeddings and generator wrappers
- `src/llm_ml_assistant/utils` - config loading utilities
- `src/llm_ml_assistant/cli.py` - CLI commands (`index`, `ask`)
- `configs/base.yaml` - runtime config
- `data/examples` - tiny demo knowledge base
- `artifacts/` - generated index files (git-ignored)
- `logs/` - runtime logs (git-ignored)
- `tests/` - automated tests

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

## Build persistent index

```bash
llm-ml-assistant index --config configs/base.yaml --data-dir data/examples --artifacts-dir artifacts --rebuild
```

What it does:
- Reads `.txt` and `.md` recursively from the data directory.
- Saves `artifacts/rag_index.faiss` and `artifacts/rag_chunks.json`.
- Writes CLI logs to `logs/cli.log`.

## Ask a question

```bash
llm-ml-assistant ask "What is RAG?" --config configs/base.yaml --artifacts-dir artifacts
```

Note: first run may download models from Hugging Face, so internet access is required.

## Alternative demo entrypoint

```bash
python -m llm_ml_assistant.main
```

## Tests

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
