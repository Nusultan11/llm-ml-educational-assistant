import unittest

from llm_ml_assistant.data.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_with_overlap(self):
        text = "abcdefghij"
        chunks = chunk_text(text, chunk_size=4, chunk_overlap=1)
        self.assertEqual(chunks, ["abcd", "defg", "ghij", "j"])

    def test_chunk_text_empty_input(self):
        self.assertEqual(chunk_text("", chunk_size=4, chunk_overlap=1), [])

    def test_chunk_text_without_overlap(self):
        text = "abcdefgh"
        chunks = chunk_text(text, chunk_size=3, chunk_overlap=0)
        self.assertEqual(chunks, ["abc", "def", "gh"])

    def test_chunk_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=0, chunk_overlap=0)

    def test_chunk_overlap_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=2, chunk_overlap=-1)

    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=2, chunk_overlap=2)


if __name__ == "__main__":
    unittest.main()
