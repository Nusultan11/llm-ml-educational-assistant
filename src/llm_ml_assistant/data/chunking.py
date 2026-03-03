from typing import List


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks