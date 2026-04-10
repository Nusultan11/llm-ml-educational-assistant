import json
import re
import tempfile
import unittest
from pathlib import Path

from llm_ml_assistant.data.cleaning import (
    clean_processed_datasets,
    score_rag_row,
    score_sft_row,
    validate_rag_row,
    validate_sft_row,
)


NOISE_PATTERN = re.compile(r"(?:http[s]?://|<[^>]+>|\bN/A\b|\blorem ipsum\b)", re.IGNORECASE)


class DataCleaningTests(unittest.TestCase):
    def test_score_rag_row_ranks_richer_document_higher(self):
        low_score, low_bucket = score_rag_row(
            {
                "source": "openassistant",
                "title": "RAG",
                "text": "RAG helps answer questions with retrieved context.",
            }
        )
        high_score, high_bucket = score_rag_row(
            {
                "source": "arxiv",
                "title": "Retrieval-Augmented Generation for Grounded QA",
                "text": (
                    "Retrieval-augmented generation combines external retrieval with generation to ground model "
                    "responses in evidence. For example, the model can retrieve supporting passages before writing "
                    "an answer, which improves factual consistency and reduces hallucinations."
                ),
            }
        )

        self.assertGreater(high_score, low_score)
        self.assertIn(low_bucket, {"low", "medium", "high"})
        self.assertEqual(high_bucket, "high")

    def test_score_sft_row_ranks_structured_answer_higher(self):
        low_score, _ = score_sft_row(
            {
                "source": "dolly",
                "instruction": "What is overfitting?",
                "response": "It is when a model learns too much.",
            }
        )
        high_score, high_bucket = score_sft_row(
            {
                "source": "dolly",
                "instruction": "Explain overfitting with an example and mitigation methods.",
                "response": (
                    "Overfitting happens when a model memorizes noise instead of learning general patterns. "
                    "For example, a model may perform perfectly on training data but fail on unseen data. "
                    "Common fixes include regularization, dropout, early stopping, and collecting more diverse data."
                ),
            }
        )

        self.assertGreater(high_score, low_score)
        self.assertEqual(high_bucket, "high")

    def test_validate_rag_row_rejects_missing_source(self):
        result = validate_rag_row(
            {
                "title": "What is RAG",
                "text": "RAG combines retrieval with generation to ground answers in external context.",
            },
            min_chars=40,
            max_non_ascii=0.30,
            noise_pattern=NOISE_PATTERN,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "missing_source")

    def test_validate_rag_row_rejects_weak_title_text_pair(self):
        result = validate_rag_row(
            {
                "source": "arxiv",
                "title": "",
                "text": "Retrieval generation helps grounding answers reliably.",
            },
            min_chars=20,
            max_non_ascii=0.30,
            noise_pattern=NOISE_PATTERN,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "weak_title_text_pair")

    def test_validate_rag_row_applies_arxiv_specific_rule(self):
        result = validate_rag_row(
            {
                "source": "arxiv",
                "title": "",
                "text": "Retrieval augmented generation combines retrieved evidence with generation to improve factual grounding in answers.",
            },
            min_chars=40,
            max_non_ascii=0.30,
            noise_pattern=NOISE_PATTERN,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "arxiv_missing_title")

    def test_validate_sft_row_rejects_weak_response(self):
        result = validate_sft_row(
            {
                "source": "openassistant",
                "instruction": "What is overfitting in machine learning?",
                "response": "yes",
            },
            min_instruction_chars=10,
            min_response_chars=2,
            max_non_ascii=0.30,
            noise_pattern=NOISE_PATTERN,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "weak_response")

    def test_validate_sft_row_rejects_invalid_instruction_response_pair(self):
        result = validate_sft_row(
            {
                "source": "openassistant",
                "instruction": "Explain gradient descent clearly.",
                "response": "Explain gradient descent clearly.",
            },
            min_instruction_chars=10,
            min_response_chars=10,
            max_non_ascii=0.30,
            noise_pattern=NOISE_PATTERN,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "invalid_instruction_response_pair")

    def test_validate_sft_row_applies_openassistant_specific_rule(self):
        result = validate_sft_row(
            {
                "source": "openassistant",
                "instruction": "Explain what gradient clipping does during training.",
                "response": "Thanks, happy to help.",
            },
            min_instruction_chars=10,
            min_response_chars=10,
            max_non_ascii=0.30,
            noise_pattern=NOISE_PATTERN,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "openassistant_weak_assistant_reply")

    def test_clean_processed_datasets_filters_noise_duplicates_and_logs_quality_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            in_dir = root / "processed"
            out_dir = root / "processed_v2_clean"
            rag_docs_dir = root / "rag_docs_v2_clean"
            audit_dir = root / "reports" / "data_audit"
            in_dir.mkdir(parents=True, exist_ok=True)

            rag_rows = [
                {
                    "id": "r1",
                    "source": "openassistant",
                    "title": "Contrastive learning",
                    "text": "Contrastive learning compares positive and negative pairs in representation space for robust embeddings.",
                    "tags": ["ml_assistant"],
                },
                {
                    "id": "r1-dup",
                    "source": "openassistant",
                    "title": "Contrastive learning",
                    "text": "Contrastive learning compares positive and negative pairs in representation space for robust embeddings.",
                    "tags": ["ml_assistant"],
                },
                {
                    "id": "r2-noise",
                    "source": "openassistant",
                    "title": "bad",
                    "text": "Visit http://spam.example for lorem ipsum",
                    "tags": ["ml_assistant"],
                },
                {
                    "id": "r3-short",
                    "source": "arxiv",
                    "title": "tiny",
                    "text": "short",
                    "tags": ["ml_assistant"],
                },
            ]
            sft_rows = [
                {
                    "id": "s1",
                    "source": "dolly",
                    "instruction": "Explain FAISS index usage in simple words",
                    "response": "FAISS stores vectors and quickly returns nearest neighbors for semantic search tasks.",
                },
                {
                    "id": "s1-dup",
                    "source": "dolly",
                    "instruction": "Explain FAISS index usage in simple words",
                    "response": "FAISS stores vectors and quickly returns nearest neighbors for semantic search tasks.",
                },
                {
                    "id": "s2-noise",
                    "source": "openassistant",
                    "instruction": "N/A",
                    "response": "http://spam",
                },
            ]

            with (in_dir / "rag_corpus.jsonl").open("w", encoding="utf-8") as f:
                for row in rag_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with (in_dir / "sft_instructions.jsonl").open("w", encoding="utf-8") as f:
                for row in sft_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            summary = clean_processed_datasets(
                in_dir=in_dir,
                out_dir=out_dir,
                rag_docs_dir=rag_docs_dir,
                audit_dir=audit_dir,
                min_rag_chars=40,
                min_instruction_chars=10,
                min_response_chars=20,
                max_non_ascii_ratio=0.30,
                noise_pattern=r"(?:http[s]?://|<[^>]+>|\bN/A\b|\blorem ipsum\b)",
                audit_sample_limit=2,
            )

            self.assertEqual(summary["rag"]["before"], 4)
            self.assertEqual(summary["rag"]["after"], 1)
            self.assertEqual(summary["sft"]["before"], 3)
            self.assertEqual(summary["sft"]["after"], 1)
            self.assertEqual(summary["rag_docs_count"], 1)
            self.assertEqual(summary["rag"]["drop_reasons"]["duplicate"], 1)
            self.assertEqual(summary["rag"]["drop_reasons"]["noise"], 1)
            self.assertEqual(summary["rag"]["drop_reasons"]["too_short_for_rag"], 1)
            self.assertEqual(summary["sft"]["drop_reasons"]["duplicate"], 1)
            self.assertEqual(summary["sft"]["drop_reasons"]["instruction_too_short"], 1)
            self.assertIn("quality_standards", summary)
            self.assertIn("rag", summary["quality_standards"])
            self.assertIn("sft", summary["quality_standards"])
            self.assertIn("quality_scoring", summary)
            self.assertIn("source_specific_rules", summary)
            self.assertIn("arxiv", summary["source_specific_rules"]["rag"])
            self.assertIn("openassistant", summary["source_specific_rules"]["sft"])
            self.assertEqual(summary["audit_dir"], str(audit_dir))
            self.assertEqual(summary["params"]["audit_sample_limit"], 2)
            self.assertIn("per_source", summary["rag"])
            self.assertIn("per_source", summary["sft"])
            self.assertEqual(summary["rag"]["per_source"]["openassistant"]["before"], 3)
            self.assertEqual(summary["rag"]["per_source"]["openassistant"]["after"], 1)
            self.assertEqual(summary["rag"]["per_source"]["openassistant"]["drop_reasons"]["duplicate"], 1)
            self.assertEqual(summary["rag"]["per_source"]["openassistant"]["drop_reasons"]["noise"], 1)
            self.assertEqual(summary["rag"]["per_source"]["arxiv"]["before"], 1)
            self.assertEqual(summary["rag"]["per_source"]["arxiv"]["removed"], 1)
            self.assertEqual(summary["rag"]["per_source"]["arxiv"]["drop_reasons"]["too_short_for_rag"], 1)
            self.assertEqual(summary["sft"]["per_source"]["dolly"]["before"], 2)
            self.assertEqual(summary["sft"]["per_source"]["dolly"]["after"], 1)
            self.assertEqual(summary["sft"]["per_source"]["dolly"]["drop_reasons"]["duplicate"], 1)
            self.assertEqual(summary["sft"]["per_source"]["openassistant"]["before"], 1)
            self.assertEqual(summary["sft"]["per_source"]["openassistant"]["removed"], 1)
            self.assertEqual(
                summary["sft"]["per_source"]["openassistant"]["drop_reasons"]["instruction_too_short"], 1
            )
            self.assertIn("quality_buckets", summary["rag"])
            self.assertIn("quality_buckets", summary["sft"])
            self.assertAlmostEqual(summary["rag"]["avg_quality_score"], 0.522, places=3)
            self.assertAlmostEqual(summary["sft"]["avg_quality_score"], 0.446, places=3)
            self.assertIsNotNone(summary["audit_sampling"])
            self.assertGreaterEqual(summary["audit_sampling"]["file_count"], 4)

            self.assertTrue((out_dir / "rag_corpus.jsonl").exists())
            self.assertTrue((out_dir / "sft_instructions.jsonl").exists())
            self.assertTrue((out_dir / "cleaning_summary.json").exists())
            self.assertEqual(len(list(rag_docs_dir.glob("*.txt"))), 1)
            self.assertTrue((audit_dir / "rag" / "rejected" / "openassistant__noise.jsonl").exists())
            self.assertTrue((audit_dir / "rag" / "kept" / "openassistant__kept_medium_quality.jsonl").exists())
            self.assertTrue((audit_dir / "sft" / "rejected" / "openassistant__instruction_too_short.jsonl").exists())
            self.assertTrue((audit_dir / "sft" / "kept" / "dolly__kept_low_quality.jsonl").exists())

            rag_rejected_sample = [
                json.loads(line)
                for line in (audit_dir / "rag" / "rejected" / "openassistant__noise.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ][0]
            self.assertEqual(rag_rejected_sample["status"], "rejected")
            self.assertEqual(rag_rejected_sample["reject_reason"], "noise")
            self.assertIn("preview", rag_rejected_sample)
            self.assertIn("normalized_length", rag_rejected_sample)

            sft_kept_sample = [
                json.loads(line)
                for line in (audit_dir / "sft" / "kept" / "dolly__kept_low_quality.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ][0]
            self.assertEqual(sft_kept_sample["status"], "kept")
            self.assertIsNone(sft_kept_sample["reject_reason"])
            self.assertIn("instruction", sft_kept_sample)
            self.assertIn("response", sft_kept_sample)
            self.assertIn("quality_score", sft_kept_sample)
            self.assertEqual(sft_kept_sample["quality_bucket"], "low")

            kept_rag_rows = [
                json.loads(line)
                for line in (out_dir / "rag_corpus.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            kept_sft_rows = [
                json.loads(line)
                for line in (out_dir / "sft_instructions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("quality_score", kept_rag_rows[0])
            self.assertIn("quality_bucket", kept_rag_rows[0])
            self.assertIn("quality_score", kept_sft_rows[0])
            self.assertIn("quality_bucket", kept_sft_rows[0])


if __name__ == "__main__":
    unittest.main()
