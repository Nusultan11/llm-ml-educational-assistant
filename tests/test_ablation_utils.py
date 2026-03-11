import unittest

from llm_ml_assistant.utils.ablation import (
    generate_retrieval_variants,
    parse_csv_ints,
    parse_csv_strings,
    safe_run_label,
)


class AblationUtilsTests(unittest.TestCase):
    def test_parse_csv_ints_deduplicates_and_preserves_order(self):
        values = parse_csv_ints("520, 700, 520, 900", "chunk_sizes")
        self.assertEqual(values, [520, 700, 900])

    def test_parse_csv_ints_rejects_invalid_value(self):
        with self.assertRaises(ValueError):
            parse_csv_ints("520,foo", "chunk_sizes")

    def test_parse_csv_strings_deduplicates_and_preserves_order(self):
        values = parse_csv_strings("hybrid, vector, hybrid", "retrieval_modes")
        self.assertEqual(values, ["hybrid", "vector"])

    def test_safe_run_label_normalizes_string(self):
        label = safe_run_label("ablation hybrid c=520/o=80 k=5")
        self.assertEqual(label, "ablation_hybrid_c_520_o_80_k_5")

    def test_generate_retrieval_variants_skips_invalid_overlap(self):
        variants = generate_retrieval_variants(
            chunk_sizes=[100],
            chunk_overlaps=[20, 100],
            top_ks=[5],
            retrieval_modes=["hybrid"],
        )
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].chunk_size, 100)
        self.assertEqual(variants[0].chunk_overlap, 20)


if __name__ == "__main__":
    unittest.main()
