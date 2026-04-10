import unittest

from llm_ml_assistant.core.context_assembler import ContextAssembler
from llm_ml_assistant.data.ingestion import ChunkRecord


class ContextAssemblerTests(unittest.TestCase):
    def test_assembler_deduplicates_and_limits_same_document(self):
        assembler = ContextAssembler(
            max_blocks=3,
            max_chars=1000,
            max_chunks_per_doc=1,
            dedup_threshold=0.75,
            expand_neighbors=False,
            chunk_size_hint=500,
        )

        records = [
            ChunkRecord(
                doc_id="doc_a",
                source_path="docs/a.txt",
                source_name="a.txt",
                title="RAG",
                chunk_id="doc_a__0000",
                text="RAG reduces hallucinations by grounding answers in retrieved context.",
                start_char=0,
                end_char=70,
                section="RAG",
            ),
            ChunkRecord(
                doc_id="doc_a",
                source_path="docs/a.txt",
                source_name="a.txt",
                title="RAG",
                chunk_id="doc_a__0001",
                text="RAG reduces hallucinations by grounding answers in retrieved context.",
                start_char=71,
                end_char=141,
                section="RAG",
            ),
            ChunkRecord(
                doc_id="doc_b",
                source_path="docs/b.txt",
                source_name="b.txt",
                title="Benefits",
                chunk_id="doc_b__0000",
                text="Retrieved evidence helps the model stay closer to source facts.",
                start_char=0,
                end_char=63,
                section="Benefits",
            ),
        ]

        contexts = assembler.assemble(records, records)

        self.assertEqual(len(contexts), 2)
        self.assertIn("Source: a.txt", contexts[0])
        self.assertIn("Source: b.txt", contexts[1])

    def test_assembler_expands_with_next_neighbor_when_chunk_looks_incomplete(self):
        assembler = ContextAssembler(
            max_blocks=2,
            max_chars=1000,
            max_chunks_per_doc=2,
            dedup_threshold=0.8,
            expand_neighbors=True,
            chunk_size_hint=500,
        )

        first = ChunkRecord(
            doc_id="doc_a",
            source_path="docs/a.txt",
            source_name="a.txt",
            title="Grounding",
            chunk_id="doc_a__0000",
            text="RAG reduces hallucinations because it grounds answers",
            start_char=0,
            end_char=53,
            section="Grounding",
            prev_chunk_id=None,
            next_chunk_id="doc_a__0001",
        )
        second = ChunkRecord(
            doc_id="doc_a",
            source_path="docs/a.txt",
            source_name="a.txt",
            title="Grounding",
            chunk_id="doc_a__0001",
            text="in retrieved context and external evidence.",
            start_char=54,
            end_char=97,
            section="Grounding",
            prev_chunk_id="doc_a__0000",
            next_chunk_id=None,
        )

        contexts = assembler.assemble([first], [first, second])

        self.assertEqual(len(contexts), 1)
        self.assertIn("grounds answers", contexts[0])
        self.assertIn("external evidence", contexts[0])

    def test_assemble_blocks_exposes_source_attribution_payload(self):
        assembler = ContextAssembler(
            max_blocks=2,
            max_chars=1000,
            max_chunks_per_doc=2,
            dedup_threshold=0.8,
            expand_neighbors=False,
            chunk_size_hint=500,
        )

        record = ChunkRecord(
            doc_id="doc_a",
            source_path="docs/rag_intro.txt",
            source_name="rag_intro.txt",
            title="What is RAG",
            chunk_id="doc_a__0003",
            text="RAG reduces hallucinations by grounding answers in retrieved context.",
            start_char=120,
            end_char=190,
            section="Grounding",
            prev_chunk_id="doc_a__0002",
            next_chunk_id="doc_a__0004",
        )

        blocks = assembler.assemble_blocks([record], [record])

        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block.rank, 1)
        self.assertEqual(block.source_name, "rag_intro.txt")
        self.assertEqual(block.chunk_ids, ["doc_a__0003"])
        self.assertEqual(block.start_char, 120)
        self.assertEqual(block.end_char, 190)
        self.assertIn("Source: rag_intro.txt", block.rendered_text)

        source = block.to_source_dict()
        self.assertEqual(source["doc_id"], "doc_a")
        self.assertEqual(source["title"], "What is RAG")
        self.assertEqual(source["section"], "Grounding")
        self.assertEqual(source["chunk_ids"], ["doc_a__0003"])
        self.assertIn("grounding answers", source["text"])


if __name__ == "__main__":
    unittest.main()
