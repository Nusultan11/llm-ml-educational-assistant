from dataclasses import dataclass
import re
from typing import List


@dataclass(frozen=True)
class ChunkSpan:
    start_char: int
    end_char: int
    text: str
    section: str | None = None


@dataclass(frozen=True)
class TextBlock:
    start_char: int
    end_char: int
    text: str
    section: str | None = None


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _is_heading_block(block_text: str) -> bool:
    stripped = block_text.strip()
    if not stripped:
        return False

    first_line = stripped.splitlines()[0].strip()
    if first_line.startswith("#"):
        return True
    if first_line.endswith(":") and len(first_line) <= 100:
        return True
    return False


def _normalize_heading(block_text: str) -> str | None:
    stripped = block_text.strip()
    if not stripped:
        return None
    first_line = stripped.splitlines()[0].strip()
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip() or None
    if first_line.endswith(":"):
        return first_line[:-1].strip() or None
    return first_line or None


def _split_text_blocks(text: str, default_section: str | None = None) -> list[TextBlock]:
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    blocks: list[TextBlock] = []
    block_start: int | None = None
    cursor = 0
    current_heading = default_section

    for line in lines:
        line_start = cursor
        line_end = cursor + len(line)
        cursor = line_end

        if line.strip():
            if block_start is None:
                block_start = line_start
            continue

        if block_start is None:
            continue

        start_char, end_char = _trim_span(text, block_start, line_start)
        if start_char < end_char:
            block_text = text[start_char:end_char]
            if _is_heading_block(block_text):
                current_heading = _normalize_heading(block_text)
                section = current_heading
            else:
                section = current_heading
            blocks.append(
                TextBlock(
                    start_char=start_char,
                    end_char=end_char,
                    text=block_text,
                    section=section,
                )
            )
        block_start = None

    if block_start is not None:
        start_char, end_char = _trim_span(text, block_start, len(text))
        if start_char < end_char:
            block_text = text[start_char:end_char]
            if _is_heading_block(block_text):
                current_heading = _normalize_heading(block_text)
                section = current_heading
            else:
                section = current_heading
            blocks.append(
                TextBlock(
                    start_char=start_char,
                    end_char=end_char,
                    text=block_text,
                    section=section,
                )
            )

    return blocks


def _merge_heading_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    merged: list[TextBlock] = []
    idx = 0

    while idx < len(blocks):
        block = blocks[idx]
        if _is_heading_block(block.text) and idx + 1 < len(blocks):
            next_block = blocks[idx + 1]
            if not _is_heading_block(next_block.text):
                merged.append(
                    TextBlock(
                        start_char=block.start_char,
                        end_char=next_block.end_char,
                        text=f"{block.text}\n\n{next_block.text}",
                        section=block.section,
                    )
                )
                idx += 2
                continue

        merged.append(block)
        idx += 1

    return merged


def _split_sentences(block: TextBlock) -> list[TextBlock]:
    sentence_pattern = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
    matches = list(sentence_pattern.finditer(block.text))
    if len(matches) <= 1:
        return [block]

    parts: list[TextBlock] = []
    for match in matches:
        raw = match.group(0)
        if not raw.strip():
            continue
        local_start, local_end = _trim_span(block.text, match.start(), match.end())
        if local_start >= local_end:
            continue
        parts.append(
            TextBlock(
                start_char=block.start_char + local_start,
                end_char=block.start_char + local_end,
                text=block.text[local_start:local_end],
                section=block.section,
            )
        )

    return parts or [block]


def _expand_long_block(block: TextBlock, chunk_size: int, chunk_overlap: int) -> list[TextBlock]:
    if len(block.text) <= chunk_size:
        return [block]

    sentence_blocks = _split_sentences(block)
    if len(sentence_blocks) > 1 and all(len(item.text) <= chunk_size for item in sentence_blocks):
        packed: list[TextBlock] = []
        current_parts: list[TextBlock] = []
        current_len = 0

        for sentence in sentence_blocks:
            addition = len(sentence.text) if not current_parts else 1 + len(sentence.text)
            if current_parts and current_len + addition > chunk_size:
                packed.append(
                    TextBlock(
                        start_char=current_parts[0].start_char,
                        end_char=current_parts[-1].end_char,
                        text=" ".join(part.text for part in current_parts),
                        section=current_parts[0].section,
                    )
                )
                current_parts = [sentence]
                current_len = len(sentence.text)
            else:
                current_parts.append(sentence)
                current_len += addition

        if current_parts:
            packed.append(
                TextBlock(
                    start_char=current_parts[0].start_char,
                    end_char=current_parts[-1].end_char,
                    text=" ".join(part.text for part in current_parts),
                    section=current_parts[0].section,
                )
            )

        if all(len(item.text) <= chunk_size for item in packed):
            return packed

    return [
        TextBlock(
            start_char=start,
            end_char=end,
            text=chunk,
            section=block.section,
        )
        for start, end, chunk in chunk_text_with_spans(block.text, chunk_size, chunk_overlap)
    ]


def _collect_overlap_start(blocks: list[TextBlock], start_idx: int, end_idx: int, chunk_overlap: int) -> int:
    if chunk_overlap <= 0 or end_idx - start_idx <= 1:
        return end_idx

    overlap_len = 0
    next_start = end_idx
    for idx in range(end_idx - 1, start_idx, -1):
        addition = len(blocks[idx].text)
        if next_start != end_idx:
            addition += 2
        overlap_len += addition
        next_start = idx
        if overlap_len >= chunk_overlap:
            break

    return next_start


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    return [chunk for _, _, chunk in chunk_text_with_spans(text, chunk_size, chunk_overlap)]


def chunk_text_with_spans(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[tuple[int, int, str]]:
    _validate_chunk_params(chunk_size, chunk_overlap)

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((start, min(end, text_length), chunk))
        start += chunk_size - chunk_overlap

    return chunks


def chunk_document_with_spans(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    default_section: str | None = None,
) -> list[ChunkSpan]:
    _validate_chunk_params(chunk_size, chunk_overlap)

    blocks = _merge_heading_blocks(_split_text_blocks(text, default_section=default_section))
    if not blocks:
        return []

    expanded_blocks: list[TextBlock] = []
    for block in blocks:
        expanded_blocks.extend(_expand_long_block(block, chunk_size, chunk_overlap))

    chunks: list[ChunkSpan] = []
    start_idx = 0

    while start_idx < len(expanded_blocks):
        current_blocks: list[TextBlock] = []
        current_len = 0
        end_idx = start_idx

        while end_idx < len(expanded_blocks):
            block = expanded_blocks[end_idx]
            addition = len(block.text) if not current_blocks else 2 + len(block.text)
            if current_blocks and current_len + addition > chunk_size:
                break
            current_blocks.append(block)
            current_len += addition
            end_idx += 1

        if not current_blocks:
            block = expanded_blocks[start_idx]
            current_blocks = [block]
            end_idx = start_idx + 1

        chunks.append(
            ChunkSpan(
                start_char=current_blocks[0].start_char,
                end_char=current_blocks[-1].end_char,
                text="\n\n".join(block.text for block in current_blocks),
                section=current_blocks[0].section,
            )
        )

        if end_idx >= len(expanded_blocks):
            break

        next_start = _collect_overlap_start(
            expanded_blocks,
            start_idx,
            end_idx,
            chunk_overlap,
        )
        start_idx = next_start if next_start > start_idx else end_idx

    return chunks
