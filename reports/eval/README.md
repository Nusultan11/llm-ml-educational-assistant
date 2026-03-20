# Manual Eval Template

Use this folder for an independent retrieval evaluation set.

## Files

- `manual_eval_template.json` - starter template (copy and edit)
- `manual_eval_v1.json` - your actual manual eval set (create this file)

## Required format

Each item must be a JSON object:

```json
{
  "query": "Your question",
  "expected_substring": "short evidence phrase expected in retrieved context"
}
```

## Workflow

1. Copy template:

```bash
copy reports\\eval\\manual_eval_template.json reports\\eval\\manual_eval_v1.json
```

2. Replace entries with your own independent questions and evidence phrases.

3. Validate file:

```bash
python scripts/validate_manual_eval.py --eval reports/eval/manual_eval_v1.json
```

4. Evaluate retrieval with this manual set:

```bash
python scripts/evaluate_artifacts_retrieval.py --config configs/colab_light.yaml --artifacts-dir artifacts --eval reports/eval/manual_eval_v1.json --tag manual_eval_v1 --out reports/retrieval_metrics/manual_eval_v1.json --history-path reports/retrieval_metrics/history.jsonl
```
