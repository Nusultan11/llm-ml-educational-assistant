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
## Ablation logs

Automated ablation runs are stored under:

- `reports/retrieval_metrics/ablation/<run_id>/run_summary.json`
- `reports/retrieval_metrics/ablation/<run_id>/leaderboard.csv`
- `reports/retrieval_metrics/ablation/<run_id>/leaderboard.md`

## Final tuned profile (Colab)

- Profile file: `reports/retrieval_metrics/final_profile_colab_20260311.json`
- Selected config: `hybrid`, `chunk_size=700`, `chunk_overlap=40`, `top_k=5`
- Selection evidence:
  - top-k run: `20260311T162804Z`
  - chunk-size run: `20260311T163355Z`
  - overlap run: `20260311T164926Z`

## Manual eval (independent)

Use a separate hand-written QA set to avoid auto-eval bias:

- Template: `reports/eval/manual_eval_template.json`
- Validator: `python scripts/validate_manual_eval.py --eval reports/eval/manual_eval_v1.json`
- Evaluation run: `python scripts/evaluate_artifacts_retrieval.py --config configs/colab_light.yaml --artifacts-dir artifacts --eval reports/eval/manual_eval_v1.json --tag manual_eval_v1 --out reports/retrieval_metrics/manual_eval_v1.json --history-path reports/retrieval_metrics/history.jsonl`
