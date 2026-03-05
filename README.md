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
- `configs/` - runtime profiles (`base`, `dev_cpu`, `colab_t4`)
- `data/examples` - tiny demo knowledge base and eval set
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

## Config profiles (with rationale)

- `configs/dev_cpu.yaml`
  - For local laptops and stable runs without GPU pressure.
  - Smaller chunks and moderate `top_k`.
- `configs/colab_t4.yaml`
  - For Google Colab T4 sessions.
  - Larger chunks and higher `top_k` for better recall.
- `configs/base.yaml`
  - Balanced default profile.

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

## Mini retrieval evaluation (to justify config choices)

```bash
python scripts/evaluate_retrieval.py --config configs/dev_cpu.yaml --data-dir data/examples --eval data/examples/eval_qa.json
python scripts/evaluate_retrieval.py --config configs/base.yaml --data-dir data/examples --eval data/examples/eval_qa.json
python scripts/evaluate_retrieval.py --config configs/colab_t4.yaml --data-dir data/examples --eval data/examples/eval_qa.json
```

Compare:
- `HitRate@k`
- `MRR@k`

Pick the profile that gives the best quality/speed trade-off for your environment.

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
