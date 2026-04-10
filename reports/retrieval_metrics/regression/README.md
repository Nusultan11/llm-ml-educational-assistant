# RAG Regression Control

This folder stores regression-check runs for the RAG pipeline.

The goal is simple: every retrieval change should leave a measurable trace.

## What a regression run does

1. Validates the eval files.
2. Runs artifact-based retrieval evaluation on the fixed auto-eval set.
3. Optionally runs the independent manual eval set.
4. Compares current metrics against baseline JSON files.
5. Writes a machine-readable summary and fails if metrics regress beyond allowed thresholds.

## Recommended command

```bash
python scripts/run_rag_regression.py \
  --config configs/colab_light.yaml \
  --artifacts-dir artifacts \
  --auto-eval data/processed_v2_clean/eval_auto_qa.json \
  --manual-eval reports/eval/manual_eval_v1.json \
  --auto-baseline reports/retrieval_metrics/v2_clean_baseline.json \
  --tag step8_check \
  --out-dir reports/retrieval_metrics/regression \
  --history-path reports/retrieval_metrics/regression/history.jsonl
```

If your embedding model is already cached locally and you want to avoid Hub calls, set:

```bash
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
```

## Output

Each run creates:

- `reports/retrieval_metrics/regression/<run_tag>/summary.json`
- `reports/retrieval_metrics/regression/<run_tag>/summary.md`
- `reports/retrieval_metrics/regression/<run_tag>/metrics/auto_eval.json`
- `reports/retrieval_metrics/regression/<run_tag>/metrics/manual_eval.json` if manual eval was enabled

## Suggested workflow

- Run this after every major retrieval change.
- Update baseline files only after reviewing a run you want to bless.
- Keep manual eval independent from auto-generated eval sets.
