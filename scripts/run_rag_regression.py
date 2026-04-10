import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_artifacts_retrieval import evaluate_artifacts_retrieval
from llm_ml_assistant.utils.eval_validation import load_eval_items, validate_eval_items
from llm_ml_assistant.utils.regression import build_regression_check


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_eval_file(
    eval_path: Path,
    min_query_chars: int,
    min_expected_chars: int,
) -> dict:
    items = load_eval_items(eval_path)
    summary, errors = validate_eval_items(
        items,
        min_query_chars=min_query_chars,
        min_expected_chars=min_expected_chars,
    )
    return {
        "path": str(eval_path),
        "summary": summary,
        "errors": errors,
        "valid": not errors,
    }


def _evaluate_named_run(
    name: str,
    config_path: Path,
    artifacts_dir: Path,
    eval_path: Path,
    tag: str,
    snapshot_label: str,
    baseline_path: Path | None,
    max_hit_rate_drop: float,
    max_mrr_drop: float,
    metrics_out: Path,
) -> dict:
    metrics = evaluate_artifacts_retrieval(
        config_path=config_path,
        artifacts_dir=artifacts_dir,
        eval_path=eval_path,
        snapshot_label=snapshot_label,
        tag=tag,
    )
    _write_json(metrics_out, metrics)

    result = {
        "name": name,
        "metrics_file": str(metrics_out),
        "metrics": metrics,
        "baseline_file": str(baseline_path) if baseline_path else "",
        "baseline_exists": bool(baseline_path and baseline_path.exists()),
    }

    if baseline_path and baseline_path.exists():
        baseline_metrics = _load_metrics(baseline_path)
        result["baseline_metrics"] = baseline_metrics
        result["regression"] = build_regression_check(
            current_metrics=metrics,
            baseline_metrics=baseline_metrics,
            max_hit_rate_drop=max_hit_rate_drop,
            max_mrr_drop=max_mrr_drop,
        )
    else:
        result["regression"] = {
            "passed": True,
            "metrics": [],
            "failure_reasons": [],
            "note": "No baseline file supplied or baseline file does not exist.",
        }

    return result


def _write_markdown_summary(path: Path, summary: dict) -> None:
    lines = [
        "# RAG Regression Summary",
        "",
        f"- Run tag: `{summary['run_tag']}`",
        f"- Created at: `{summary['created_at_utc']}`",
        f"- Overall passed: `{summary['overall_passed']}`",
        f"- Config: `{summary['config']}`",
        f"- Artifacts dir: `{summary['artifacts_dir']}`",
        "",
        "## Eval validation",
        "",
    ]

    for eval_check in summary["eval_validation"]:
        lines.extend(
            [
                f"### `{eval_check['path']}`",
                f"- Valid: `{eval_check['valid']}`",
                f"- Items: `{eval_check['summary']['items']}`",
                f"- Unique queries: `{eval_check['summary']['unique_queries']}`",
                f"- Errors: `{eval_check['summary']['errors']}`",
                "",
            ]
        )
        if eval_check["errors"]:
            lines.append("Errors:")
            for err in eval_check["errors"]:
                lines.append(f"- {err}")
            lines.append("")

    lines.extend(["## Regression checks", ""])
    for run in summary["runs"]:
        lines.extend(
            [
                f"### `{run['name']}`",
                f"- Metrics file: `{run['metrics_file']}`",
                f"- HitRate@{run['metrics']['top_k']}: `{run['metrics']['hit_rate']:.3f}`",
                f"- MRR@{run['metrics']['top_k']}: `{run['metrics']['mrr']:.3f}`",
                f"- Baseline exists: `{run['baseline_exists']}`",
                f"- Passed: `{run['regression']['passed']}`",
            ]
        )
        for metric in run["regression"].get("metrics", []):
            lines.append(
                f"- {metric['metric']}: current={metric['current']:.3f}, "
                f"baseline={metric['baseline']:.3f}, delta={metric['delta']:+.3f}"
            )
        for reason in run["regression"].get("failure_reasons", []):
            lines.append(f"- Failure: {reason}")
        if run["regression"].get("note"):
            lines.append(f"- Note: {run['regression']['note']}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _persist_summary(run_dir: Path, summary: dict, history_path: Path) -> tuple[Path, Path]:
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    _write_json(summary_path, summary)
    _write_markdown_summary(summary_md_path, summary)
    _append_jsonl(history_path, summary)
    return summary_path, summary_md_path


def main(
    config_path: Path = typer.Option(Path("configs/colab_light.yaml"), "--config"),
    artifacts_dir: Path = typer.Option(Path("artifacts"), "--artifacts-dir"),
    auto_eval_path: Path = typer.Option(Path("data/processed_v2_clean/eval_auto_qa.json"), "--auto-eval"),
    manual_eval_path: Path = typer.Option(Path("reports/eval/manual_eval_v1.json"), "--manual-eval"),
    auto_baseline_path: Path | None = typer.Option(
        Path("reports/retrieval_metrics/v2_clean_baseline.json"),
        "--auto-baseline",
    ),
    manual_baseline_path: Path | None = typer.Option(
        None,
        "--manual-baseline",
        help="Optional baseline metrics JSON for manual eval.",
    ),
    tag: str = typer.Option("rag_regression", "--tag"),
    snapshot_label: str = typer.Option("", "--snapshot-label"),
    out_dir: Path = typer.Option(Path("reports/retrieval_metrics/regression"), "--out-dir"),
    history_path: Path = typer.Option(
        Path("reports/retrieval_metrics/regression/history.jsonl"),
        "--history-path",
    ),
    max_hit_rate_drop: float = typer.Option(0.0, "--max-hit-rate-drop"),
    max_mrr_drop: float = typer.Option(0.0, "--max-mrr-drop"),
    skip_manual: bool = typer.Option(False, "--skip-manual"),
    require_manual: bool = typer.Option(False, "--require-manual"),
    min_query_chars: int = typer.Option(8, "--min-query-chars"),
    min_expected_chars: int = typer.Option(8, "--min-expected-chars"),
):
    if not auto_eval_path.exists():
        raise FileNotFoundError(f"Auto eval file not found: {auto_eval_path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_tag = f"{tag}_{run_id}"
    run_dir = out_dir / run_tag
    metrics_dir = run_dir / "metrics"

    eval_validation = [
        _validate_eval_file(
            auto_eval_path,
            min_query_chars=min_query_chars,
            min_expected_chars=min_expected_chars,
        )
    ]

    manual_eval_enabled = not skip_manual and manual_eval_path.exists()
    if not skip_manual and not manual_eval_path.exists():
        if require_manual:
            summary = {
                "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "run_tag": run_tag,
                "config": str(config_path),
                "artifacts_dir": str(artifacts_dir),
                "overall_passed": False,
                "eval_validation": eval_validation,
                "runs": [],
                "failure_reasons": [f"Manual eval file not found: {manual_eval_path}"],
            }
            summary_path, summary_md_path = _persist_summary(run_dir, summary, history_path)
            print(f"Summary JSON: {summary_path}")
            print(f"Summary MD: {summary_md_path}")
            raise typer.Exit(code=1)

    if manual_eval_enabled:
        eval_validation.append(
            _validate_eval_file(
                manual_eval_path,
                min_query_chars=min_query_chars,
                min_expected_chars=min_expected_chars,
            )
        )

    invalid = [item for item in eval_validation if not item["valid"]]
    if invalid:
        summary = {
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_tag": run_tag,
            "config": str(config_path),
            "artifacts_dir": str(artifacts_dir),
            "overall_passed": False,
            "eval_validation": eval_validation,
            "runs": [],
            "failure_reasons": ["Eval validation failed."],
        }
        summary_path, summary_md_path = _persist_summary(run_dir, summary, history_path)
        print(f"Summary JSON: {summary_path}")
        print(f"Summary MD: {summary_md_path}")
        raise typer.Exit(code=1)

    runs = [
        _evaluate_named_run(
            name="auto_eval",
            config_path=config_path,
            artifacts_dir=artifacts_dir,
            eval_path=auto_eval_path,
            tag=f"{tag}_auto",
            snapshot_label=snapshot_label,
            baseline_path=auto_baseline_path,
            max_hit_rate_drop=max_hit_rate_drop,
            max_mrr_drop=max_mrr_drop,
            metrics_out=metrics_dir / "auto_eval.json",
        )
    ]

    if manual_eval_enabled:
        runs.append(
            _evaluate_named_run(
                name="manual_eval",
                config_path=config_path,
                artifacts_dir=artifacts_dir,
                eval_path=manual_eval_path,
                tag=f"{tag}_manual",
                snapshot_label=snapshot_label,
                baseline_path=manual_baseline_path,
                max_hit_rate_drop=max_hit_rate_drop,
                max_mrr_drop=max_mrr_drop,
                metrics_out=metrics_dir / "manual_eval.json",
            )
        )

    failure_reasons: list[str] = []
    for run in runs:
        failure_reasons.extend(run["regression"].get("failure_reasons", []))

    overall_passed = not failure_reasons
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_tag": run_tag,
        "config": str(config_path),
        "artifacts_dir": str(artifacts_dir),
        "overall_passed": overall_passed,
        "eval_validation": eval_validation,
        "runs": runs,
        "failure_reasons": failure_reasons,
    }

    summary_path, summary_md_path = _persist_summary(run_dir, summary, history_path)

    print(f"Regression run: {run_tag}")
    print(f"Overall passed: {overall_passed}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary MD: {summary_md_path}")

    for run in runs:
        metrics = run["metrics"]
        print(
            f"- {run['name']}: HitRate@{metrics['top_k']}={metrics['hit_rate']:.3f}, "
            f"MRR@{metrics['top_k']}={metrics['mrr']:.3f}, passed={run['regression']['passed']}"
        )

    if not overall_passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
