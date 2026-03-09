# Retrieval Metrics Log

This folder stores reproducible retrieval evaluation results for local artifacts.

## Files

- `v2_clean_baseline.json` - latest baseline result for cleaned real-data index.
- `history.jsonl` - append-only run history (one JSON object per line).

## Current baseline (v2_clean)

- Snapshot: `20260309T152715Z__v2_clean_local_retry`
- Config: `configs/colab_light.yaml`
- Eval: `data/processed_v2_clean/eval_auto_qa.json`
- Queries: `200`
- HitRate@5: `0.330`
- MRR@5: `0.298`

## How to log a new run

```bash
python scripts/evaluate_artifacts_retrieval.py \
  --config configs/colab_light.yaml \
  --artifacts-dir artifacts \
  --eval data/processed_v2_clean/eval_auto_qa.json \
  --snapshot-label <snapshot_label> \
  --tag <run_tag> \
  --out reports/retrieval_metrics/<run_tag>.json \
  --history-path reports/retrieval_metrics/history.jsonl
```