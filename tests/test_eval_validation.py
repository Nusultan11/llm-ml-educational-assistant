import json
import tempfile
import unittest
from pathlib import Path

from llm_ml_assistant.utils.eval_validation import load_eval_items, validate_eval_items


class EvalValidationTests(unittest.TestCase):
    def test_validate_eval_items_accepts_valid_input(self):
        items = [
            {
                "query": "What is RAG?",
                "expected_substring": "retrieval-augmented generation",
            },
            {
                "query": "What is MRR?",
                "expected_substring": "relevant result appears earlier",
            },
        ]

        summary, errors = validate_eval_items(items, min_query_chars=5, min_expected_chars=10)

        self.assertEqual(summary["items"], 2)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(errors, [])

    def test_validate_eval_items_detects_issues(self):
        items = [
            {"query": "Hi", "expected_substring": "ok"},
            {"query": "Hi", "expected_substring": "ok"},
        ]

        summary, errors = validate_eval_items(items, min_query_chars=5, min_expected_chars=5)

        self.assertGreaterEqual(summary["errors"], 3)
        self.assertTrue(any("query is too short" in e for e in errors))
        self.assertTrue(any("expected_substring is too short" in e for e in errors))
        self.assertTrue(any("duplicate query" in e for e in errors))

    def test_load_eval_items_rejects_non_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"query": "x"}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_eval_items(path)


if __name__ == "__main__":
    unittest.main()
