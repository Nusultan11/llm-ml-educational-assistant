import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AssembledContextBlock:
    rank: int
    doc_id: str
    source_path: str
    source_name: str
    title: str
    section: str | None
    chunk_ids: list[str]
    start_char: int
    end_char: int
    text: str
    rendered_text: str

    def to_source_dict(self) -> dict:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "title": self.title,
            "section": self.section,
            "chunk_ids": self.chunk_ids,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
        }


class ContextAssembler:
    def __init__(
        self,
        max_blocks: int,
        max_chars: int,
        max_chunks_per_doc: int,
        dedup_threshold: float,
        expand_neighbors: bool,
        chunk_size_hint: int,
    ):
        self.max_blocks = max_blocks
        self.max_chars = max_chars
        self.max_chunks_per_doc = max_chunks_per_doc
        self.dedup_threshold = dedup_threshold
        self.expand_neighbors = expand_neighbors
        self.chunk_size_hint = chunk_size_hint

    def assemble(self, candidate_records: list, all_chunk_records: list | None = None) -> list[str]:
        return [
            block.rendered_text
            for block in self.assemble_blocks(candidate_records, all_chunk_records)
        ]

    def assemble_blocks(
        self,
        candidate_records: list,
        all_chunk_records: list | None = None,
    ) -> list[AssembledContextBlock]:
        if not candidate_records:
            return []

        chunk_map = {
            record.chunk_id: record
            for record in (all_chunk_records or candidate_records)
        }
        selected_blocks: list[AssembledContextBlock] = []
        selected_token_sets: list[set[str]] = []
        per_doc_counts: dict[str, int] = {}
        total_chars = 0

        for record in candidate_records:
            if len(selected_blocks) >= self.max_blocks:
                break

            if per_doc_counts.get(record.doc_id, 0) >= self.max_chunks_per_doc:
                continue

            block_records = self._assemble_block_records(record, chunk_map)
            block = self._build_block(len(selected_blocks) + 1, block_records)
            block_tokens = set(self._tokenize(block.text))

            if self._is_duplicate(block_tokens, selected_token_sets):
                continue

            if total_chars + len(block.rendered_text) > self.max_chars and selected_blocks:
                continue

            selected_blocks.append(block)
            selected_token_sets.append(block_tokens)
            total_chars += len(block.rendered_text)

            for block_record in block_records:
                per_doc_counts[block_record.doc_id] = per_doc_counts.get(block_record.doc_id, 0) + 1

        return selected_blocks

    def _assemble_block_records(self, record, chunk_map: dict) -> list:
        block_records = [record]

        if not self.expand_neighbors:
            return block_records

        next_record = chunk_map.get(record.next_chunk_id) if record.next_chunk_id else None
        if next_record is None:
            return block_records

        if next_record.doc_id != record.doc_id:
            return block_records

        if next_record.section != record.section:
            return block_records

        combined_len = len(record.text) + len(next_record.text)
        if combined_len > int(self.chunk_size_hint * 1.6):
            return block_records

        if self._looks_incomplete(record.text):
            block_records.append(next_record)

        return block_records

    def _looks_incomplete(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.endswith((".", "!", "?", ":", "`")):
            return False
        return True

    def _build_block(self, rank: int, block_records: list) -> AssembledContextBlock:
        first = block_records[0]
        lines = [f"Source: {first.source_name or first.doc_id}"]

        if first.title:
            lines.append(f"Title: {first.title}")
        if first.section and first.section != first.title:
            lines.append(f"Section: {first.section}")

        text_body = "\n\n".join(record.text.strip() for record in block_records if record.text.strip())
        lines.append("Text:")
        lines.append(text_body)
        return AssembledContextBlock(
            rank=rank,
            doc_id=first.doc_id,
            source_path=first.source_path,
            source_name=first.source_name or first.doc_id,
            title=first.title,
            section=first.section,
            chunk_ids=[record.chunk_id for record in block_records],
            start_char=min(record.start_char for record in block_records),
            end_char=max(record.end_char for record in block_records),
            text=text_body,
            rendered_text="\n".join(lines).strip(),
        )

    def _is_duplicate(self, candidate_tokens: set[str], selected_token_sets: list[set[str]]) -> bool:
        if not candidate_tokens:
            return False

        for existing_tokens in selected_token_sets:
            similarity = self._jaccard(candidate_tokens, existing_tokens)
            if similarity >= self.dedup_threshold:
                return True
        return False

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())
