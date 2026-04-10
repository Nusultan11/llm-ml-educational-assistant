import unittest

from llm_ml_assistant.utils.split_integrity import validate_split_integrity


class SplitIntegrityTests(unittest.TestCase):
    def test_detects_direct_sft_eval_leakage(self):
        sft_rows = [
            {
                "id": "s1",
                "source": "openassistant",
                "instruction": "What is RAG?",
                "response": "RAG combines retrieval with generation to ground answers in external context.",
            }
        ]
        auto_eval_items = [
            {
                "query": "What is RAG?",
                "expected_substring": "retrieval with generation",
            }
        ]

        summary, errors = validate_split_integrity(
            sft_rows=sft_rows,
            auto_eval_items=auto_eval_items,
            max_examples=5,
        )

        self.assertGreater(summary["checks"]["sft_to_auto_eval"]["matches"], 0)
        self.assertTrue(any("SFT split" in error for error in errors))

    def test_detects_rag_eval_leakage(self):
        rag_rows = [
            {
                "id": "r1",
                "source": "arxiv",
                "title": "What is Retrieval-Augmented Generation",
                "text": "RAG combines retrieval with generation to improve factual grounding in model answers.",
            }
        ]
        manual_eval_items = [
            {
                "query": "What is retrieval-augmented generation?",
                "expected_substring": "retrieval with generation",
            }
        ]

        summary, errors = validate_split_integrity(
            rag_rows=rag_rows,
            manual_eval_items=manual_eval_items,
            max_examples=5,
        )

        self.assertGreater(summary["checks"]["rag_to_manual_eval"]["matches"], 0)
        self.assertTrue(any("RAG corpus" in error for error in errors))

    def test_detects_overlap_between_auto_and_manual_eval(self):
        auto_eval_items = [
            {
                "query": "What is MRR?",
                "expected_substring": "ranking metric",
            }
        ]
        manual_eval_items = [
            {
                "query": "What is MRR?",
                "expected_substring": "ranking metric",
            }
        ]

        summary, errors = validate_split_integrity(
            auto_eval_items=auto_eval_items,
            manual_eval_items=manual_eval_items,
            max_examples=5,
        )

        self.assertGreater(summary["checks"]["auto_eval_to_manual_eval"]["matches"], 0)
        self.assertTrue(any("auto_eval item #1 overlaps" in error for error in errors))

    def test_passes_cleanly_when_splits_are_independent(self):
        rag_rows = [
            {
                "id": "r1",
                "source": "arxiv",
                "title": "Vector search",
                "text": "Approximate nearest neighbor search uses vector indexes for efficient retrieval.",
            }
        ]
        sft_rows = [
            {
                "id": "s1",
                "source": "dolly",
                "instruction": "Explain gradient descent.",
                "response": "Gradient descent updates model parameters by moving along the negative gradient of the loss.",
            }
        ]
        auto_eval_items = [
            {
                "query": "What does batching do during training?",
                "expected_substring": "group multiple examples",
            }
        ]
        manual_eval_items = [
            {
                "query": "Why are embeddings useful for retrieval?",
                "expected_substring": "convert text into vectors",
            }
        ]

        summary, errors = validate_split_integrity(
            rag_rows=rag_rows,
            sft_rows=sft_rows,
            auto_eval_items=auto_eval_items,
            manual_eval_items=manual_eval_items,
            max_examples=5,
        )

        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
