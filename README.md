# llm-ml-assistant

Minimal RAG assistant skeleton with local retrieval and LLM generation.

## Features

- Config loading via YAML + Pydantic
- Document chunking and vector indexing (FAISS)
- Hybrid retrieval (`vector + keyword`) with RRF fusion
- Retrieval + prompt building
- Generator wrapper for Hugging Face Transformers
- CLI commands for indexing and question answering
- Persistent index save/load (`.faiss` + chunks JSON)
- Unit and integration tests

## Repository structure

- `src/llm_ml_assistant/core` - retrieval, vector store, keyword index, prompt builder
- `src/llm_ml_assistant/models` - embeddings and generator wrappers
- `src/llm_ml_assistant/utils` - config loading utilities
- `src/llm_ml_assistant/cli.py` - CLI commands (`index`, `ask`)
- `configs/` - runtime profiles (`base`, `dev_cpu`, `colab_t4`, `colab_light`)
- `data/examples` - demo knowledge base and eval set
- `data/processed` - prepared RAG/SFT datasets
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

## Prepare datasets (RAG + SFT)

```bash
python scripts/prepare_datasets.py --out-dir data/processed
```

Outputs:
- `data/processed/rag_corpus.jsonl`
- `data/processed/sft_instructions.jsonl`
- `data/processed/summary.json`

Optional local sources (JSONL):
- `data/raw/stackoverflow.jsonl` with fields: `id,title,question,answer`
- `data/raw/arxiv.jsonl` with fields: `id,title,abstract`

## Clean datasets after EDA (local v2)

Build cleaned outputs and local RAG docs:

```bash
python scripts/clean_processed_datasets.py --in-dir data/processed --out-dir data/processed_v2_clean --rag-docs-dir data/rag_docs_v2_clean
```

Outputs:
- `data/processed_v2_clean/rag_corpus.jsonl`
- `data/processed_v2_clean/sft_instructions.jsonl`
- `data/processed_v2_clean/cleaning_summary.json`
- `data/rag_docs_v2_clean/*.txt`

## Run full local pipeline (one command)

```bash
python scripts/run_local_pipeline.py --config configs/colab_light.yaml --snapshot-label v2_clean_local
```

On Windows, keep `--artifacts-dir` and `--data-dir` relative (for example `artifacts`, `data/rag_docs_v2_clean`) to avoid FAISS path issues with Unicode absolute paths.

Default flow:
- prepare datasets (`data/processed`)
- clean datasets (`data/processed_v2_clean`)
- rebuild index from `data/rag_docs_v2_clean`
- archive artifacts snapshot

Useful flags:
- `--skip-prepare`, `--skip-clean`, `--skip-index`, `--skip-archive`
- `--no-openassistant`, `--no-dolly`
- cleaning thresholds: `--min-rag-chars`, `--min-instruction-chars`, `--min-response-chars`

## Config profiles (with rationale)

- `configs/dev_cpu.yaml`
  - Stable local runs on CPU.
  - Embedding model: `intfloat/e5-base`
  - Retrieval mode: `hybrid`
- `configs/colab_t4.yaml`
  - Colab T4-oriented profile with `Mistral-7B`.
  - Embedding model: `BAAI/bge-base-en`
  - Retrieval mode: `hybrid`
- `configs/colab_light.yaml`
  - Colab free-friendly profile with lighter generator (`Phi-3-mini`).
  - Good default when `Mistral-7B` is too heavy.
- `configs/base.yaml`
  - Balanced default profile.

## Retrieval settings

In `rag` config section:

- `retrieval_mode`: `vector` or `hybrid`
- `rrf_k`: Reciprocal Rank Fusion smoothing constant (used in `hybrid`)

## Build persistent index

```bash
llm-ml-assistant index --config configs/base.yaml --data-dir data/examples --artifacts-dir artifacts --rebuild
```

What it does:
- Reads `.txt` and `.md` recursively from the data directory.
- Saves `artifacts/rag_index.faiss` and `artifacts/rag_chunks.json`.
- Writes CLI logs to `logs/cli.log`.

## Artifact snapshots (recommended)

To preserve step-by-step index history with metadata:

```bash
python scripts/archive_artifacts.py --artifacts-dir artifacts --config configs/colab_t4.yaml --data-dir data/examples --label step_01 --notes "baseline after hybrid indexing"
```

This creates:
- `artifacts/snapshots/<timestamp>__<label>/rag_index.faiss`
- `artifacts/snapshots/<timestamp>__<label>/rag_chunks.json`
- `artifacts/snapshots/<timestamp>__<label>/metadata.json`

Metadata includes config values, chunk count, file hashes, and notes.

## Ask a question

RAG mode (default):

```bash
llm-ml-assistant ask "What is RAG?" --config configs/colab_t4.yaml --artifacts-dir artifacts --mode rag --show-contexts
```

Retrieval-only mode (no LLM, fully free/fallback):

```bash
llm-ml-assistant ask "What is RAG?" --config configs/colab_t4.yaml --artifacts-dir artifacts --mode retrieval_only --show-contexts
```

For Colab free stability:

```bash
llm-ml-assistant ask "What is RAG?" --config configs/colab_light.yaml --artifacts-dir artifacts --mode rag --show-contexts
```

Notes:
- If `--mode rag` fails (OOM/GPU/network), CLI falls back to retrieval-only and prints contexts.
- `--show-contexts` is useful for debugging and source transparency.

## Mini retrieval evaluation

```bash
python scripts/evaluate_retrieval.py --config configs/dev_cpu.yaml --data-dir data/examples --eval data/examples/eval_qa.json
python scripts/evaluate_retrieval.py --config configs/base.yaml --data-dir data/examples --eval data/examples/eval_qa.json
python scripts/evaluate_retrieval.py --config configs/colab_t4.yaml --data-dir data/examples --eval data/examples/eval_qa.json
```

Compare:
- `HitRate@k`
- `MRR@k`

For artifact-based local baseline logging (real-data index):

```bash
python scripts/evaluate_artifacts_retrieval.py --config configs/colab_light.yaml --artifacts-dir artifacts --eval data/processed_v2_clean/eval_auto_qa.json --snapshot-label <snapshot_label> --tag <run_tag> --out reports/retrieval_metrics/<run_tag>.json --history-path reports/retrieval_metrics/history.jsonl
```

See logged metrics in `reports/retrieval_metrics/`.

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




