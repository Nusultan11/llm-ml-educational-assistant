import unittest

from llm_ml_assistant.utils.regression import build_regression_check, compare_metric


class RegressionUtilsTests(unittest.TestCase):
    def test_compare_metric_passes_when_drop_within_tolerance(self):
        result = compare_metric(
            current=0.325,
            baseline=0.33,
            metric="hit_rate",
            max_drop=0.01,
        )

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.delta, -0.005)

    def test_compare_metric_fails_when_drop_exceeds_tolerance(self):
        result = compare_metric(
            current=0.29,
            baseline=0.33,
            metric="mrr",
            max_drop=0.01,
        )

        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.delta, -0.04)

    def test_build_regression_check_collects_failure_reasons(self):
        result = build_regression_check(
            current_metrics={"tag": "new", "top_k": 5, "hit_rate": 0.31, "mrr": 0.27},
            baseline_metrics={"tag": "base", "top_k": 5, "hit_rate": 0.33, "mrr": 0.30},
            max_hit_rate_drop=0.01,
            max_mrr_drop=0.01,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["current_tag"], "new")
        self.assertEqual(result["baseline_tag"], "base")
        self.assertEqual(len(result["metrics"]), 2)
        self.assertEqual(len(result["failure_reasons"]), 2)


if __name__ == "__main__":
    unittest.main()
