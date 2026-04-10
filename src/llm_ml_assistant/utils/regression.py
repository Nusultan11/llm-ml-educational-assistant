from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    current: float
    baseline: float
    delta: float
    max_drop: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "current": round(self.current, 6),
            "baseline": round(self.baseline, 6),
            "delta": round(self.delta, 6),
            "max_drop": round(self.max_drop, 6),
            "passed": self.passed,
        }


def compare_metric(
    current: float,
    baseline: float,
    metric: str,
    max_drop: float,
) -> MetricDelta:
    delta = float(current) - float(baseline)
    return MetricDelta(
        metric=metric,
        current=float(current),
        baseline=float(baseline),
        delta=delta,
        max_drop=float(max_drop),
        passed=delta >= (-1.0 * float(max_drop)),
    )


def build_regression_check(
    current_metrics: dict,
    baseline_metrics: dict,
    max_hit_rate_drop: float,
    max_mrr_drop: float,
) -> dict:
    hit_rate = compare_metric(
        current=current_metrics.get("hit_rate", 0.0),
        baseline=baseline_metrics.get("hit_rate", 0.0),
        metric="hit_rate",
        max_drop=max_hit_rate_drop,
    )
    mrr = compare_metric(
        current=current_metrics.get("mrr", 0.0),
        baseline=baseline_metrics.get("mrr", 0.0),
        metric="mrr",
        max_drop=max_mrr_drop,
    )

    passed = hit_rate.passed and mrr.passed
    failure_reasons = []
    if not hit_rate.passed:
        failure_reasons.append(
            f"hit_rate dropped by {abs(hit_rate.delta):.6f}, allowed {hit_rate.max_drop:.6f}"
        )
    if not mrr.passed:
        failure_reasons.append(
            f"mrr dropped by {abs(mrr.delta):.6f}, allowed {mrr.max_drop:.6f}"
        )

    return {
        "passed": passed,
        "current_tag": current_metrics.get("tag", ""),
        "baseline_tag": baseline_metrics.get("tag", ""),
        "top_k": current_metrics.get("top_k", baseline_metrics.get("top_k")),
        "metrics": [hit_rate.to_dict(), mrr.to_dict()],
        "failure_reasons": failure_reasons,
    }
