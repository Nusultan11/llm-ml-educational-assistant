import json
import tempfile
import unittest
from pathlib import Path

from llm_ml_assistant.data.cleaning import clean_processed_datasets


class DataCleaningTests(unittest.TestCase):
    def test_clean_processed_datasets_filters_noise_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            in_dir = root / "processed"
            out_dir = root / "processed_v2_clean"
            rag_docs_dir = root / "rag_docs_v2_clean"
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
                    "source": "openassistant",
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
                    "source": "dolly",
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
                min_rag_chars=40,
                min_instruction_chars=10,
                min_response_chars=20,
                max_non_ascii_ratio=0.30,
                noise_pattern=r"(?:http[s]?://|<[^>]+>|\\bN/A\\b|\\blorem ipsum\\b)",
            )

            self.assertEqual(summary["rag"]["before"], 4)
            self.assertEqual(summary["rag"]["after"], 1)
            self.assertEqual(summary["sft"]["before"], 3)
            self.assertEqual(summary["sft"]["after"], 1)
            self.assertEqual(summary["rag_docs_count"], 1)

            self.assertTrue((out_dir / "rag_corpus.jsonl").exists())
            self.assertTrue((out_dir / "sft_instructions.jsonl").exists())
            self.assertTrue((out_dir / "cleaning_summary.json").exists())
            self.assertEqual(len(list(rag_docs_dir.glob("*.txt"))), 1)


if __name__ == "__main__":
    unittest.main()
