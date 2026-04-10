import json
import tempfile
import unittest
from pathlib import Path

from llm_ml_assistant.utils.data_split import build_holdout_split, split_sft_rows
from llm_ml_assistant.utils.split_integrity import validate_split_integrity


class DataSplitTests(unittest.TestCase):
    def test_split_sft_rows_is_deterministic_and_preserves_totals(self):
        rows = [
            {
                "id": f"oa-{i}",
                "source": "openassistant",
                "instruction": f"q{i}",
                "response": f"r{i} long enough",
                "quality_bucket": "high" if i < 5 else "low",
            }
            for i in range(10)
        ] + [
            {
                "id": f"d-{i}",
                "source": "dolly",
                "instruction": f"dq{i}",
                "response": f"dr{i} long enough",
                "quality_bucket": "medium",
            }
            for i in range(10)
        ]

        split_a, summary_a = split_sft_rows(rows, train_ratio=0.7, dev_ratio=0.15, eval_ratio=0.15, seed=7)
        split_b, summary_b = split_sft_rows(rows, train_ratio=0.7, dev_ratio=0.15, eval_ratio=0.15, seed=7)

        self.assertEqual(summary_a["total_rows"], 20)
        self.assertLess(summary_a["train_rows"] + summary_a["dev_rows"] + summary_a["eval_holdout_rows"], 20)
        self.assertEqual(summary_a["raw_train_rows"] + summary_a["raw_dev_rows"] + summary_a["raw_eval_holdout_rows"], 20)
        self.assertEqual(summary_a, summary_b)
        self.assertEqual(
            [row["id"] for row in split_a["eval_holdout"]],
            [row["id"] for row in split_b["eval_holdout"]],
        )
        self.assertIn("quality_policy_summary", summary_a)
        self.assertIn("quality_bucket_totals", summary_a)

    def test_build_holdout_split_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sft_path = root / "sft_instructions.jsonl"
            out_dir = root / "splits"

            rows = []
            for i in range(12):
                rows.append(
                    {
                        "id": f"oa-{i}",
                        "source": "openassistant",
                        "instruction": f"Explain concept {i}",
                        "response": f"Detailed response {i} with enough text to remain useful for evaluation.",
                        "quality_bucket": "high" if i < 4 else "medium" if i < 8 else "low",
                    }
                )

            with sft_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            summary = build_holdout_split(
                sft_path=sft_path,
                out_dir=out_dir,
                train_ratio=0.75,
                dev_ratio=0.0,
                eval_ratio=0.25,
                seed=11,
            )

            self.assertTrue((out_dir / "sft_train.jsonl").exists())
            self.assertTrue((out_dir / "sft_dev.jsonl").exists())
            self.assertTrue((out_dir / "sft_eval_holdout.jsonl").exists())
            self.assertTrue((out_dir / "split_summary.json").exists())
            self.assertEqual(summary["raw_train_rows"] + summary["raw_dev_rows"] + summary["raw_eval_holdout_rows"], 12)
            self.assertLess(summary["train_rows"] + summary["dev_rows"] + summary["eval_holdout_rows"], 12)
            self.assertEqual(summary["per_source"]["openassistant"]["total"], 12)
            self.assertIn("quality_policy", summary)

    def test_holdout_eval_can_pass_split_integrity_against_train_pool(self):
        rows = [
            {
                "id": f"d-{i}",
                "source": "dolly",
                "instruction": f"Explain retrieval concept {i}",
                "response": f"Detailed answer {i} with unique wording about retrieval and grounding behavior.",
                "quality_bucket": "high" if i < 10 else "medium",
            }
            for i in range(20)
        ]

        split_rows, _ = split_sft_rows(rows, train_ratio=0.8, dev_ratio=0.1, eval_ratio=0.1, seed=19)
        eval_holdout = [
            {
                "query": row["instruction"],
                "expected_substring": row["response"][:60].strip(),
            }
            for row in split_rows["eval_holdout"]
        ]

        summary, errors = validate_split_integrity(
            sft_rows=split_rows["train"],
            auto_eval_items=eval_holdout,
            max_examples=5,
        )

        self.assertEqual(summary["checks"]["sft_to_auto_eval"]["matches"], 0)
        self.assertEqual(errors, [])

    def test_quality_policy_downweights_low_quality_rows(self):
        rows = [
            {
                "id": f"high-{i}",
                "source": "openassistant",
                "instruction": f"High question {i}",
                "response": f"High quality answer {i} with detailed explanation.",
                "quality_bucket": "high",
            }
            for i in range(8)
        ] + [
            {
                "id": f"low-{i}",
                "source": "openassistant",
                "instruction": f"Low question {i}",
                "response": f"Low quality answer {i} with enough text.",
                "quality_bucket": "low",
            }
            for i in range(8)
        ]

        split_rows, summary = split_sft_rows(
            rows,
            train_ratio=0.75,
            dev_ratio=0.0,
            eval_ratio=0.25,
            seed=5,
        )

        kept_train_buckets = [row["quality_bucket"] for row in split_rows["train"]]
        self.assertGreater(kept_train_buckets.count("high"), kept_train_buckets.count("low"))
        self.assertEqual(summary["quality_policy_summary"]["train"]["per_bucket_before"]["low"], 6)
        self.assertLess(
            summary["quality_policy_summary"]["train"]["per_bucket_after"]["low"],
            summary["quality_policy_summary"]["train"]["per_bucket_before"]["low"],
        )


if __name__ == "__main__":
    unittest.main()
