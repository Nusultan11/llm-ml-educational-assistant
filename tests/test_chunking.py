import unittest

from llm_ml_assistant.data.chunking import (
    chunk_document_with_spans,
    chunk_text,
    chunk_text_with_spans,
)


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

    def test_chunk_text_with_spans(self):
        text = "abcdefgh"
        chunks = chunk_text_with_spans(text, chunk_size=3, chunk_overlap=1)
        self.assertEqual(
            chunks,
            [
                (0, 3, "abc"),
                (2, 5, "cde"),
                (4, 7, "efg"),
                (6, 8, "gh"),
            ],
        )

    def test_chunk_document_with_spans_preserves_paragraph_boundaries(self):
        text = (
            "# What is RAG\n\n"
            "RAG combines retrieval with generation to ground answers.\n\n"
            "# Benefits\n\n"
            "It reduces hallucinations by using external evidence."
        )

        chunks = chunk_document_with_spans(text, chunk_size=90, chunk_overlap=0)

        self.assertEqual(len(chunks), 2)
        self.assertIn("What is RAG", chunks[0].text)
        self.assertIn("RAG combines retrieval", chunks[0].text)
        self.assertIn("Benefits", chunks[1].text)
        self.assertIn("reduces hallucinations", chunks[1].text)
        self.assertEqual(chunks[0].section, "What is RAG")
        self.assertEqual(chunks[1].section, "Benefits")

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
