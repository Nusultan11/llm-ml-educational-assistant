from dataclasses import asdict, dataclass
import re
from pathlib import Path

from llm_ml_assistant.data.chunking import chunk_document_with_spans


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    source_path: str
    source_name: str
    title: str
    text: str
    section: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    source_path: str
    source_name: str
    title: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    section: str | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def infer_title(text: str, fallback: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback
        return line[:120]
    return fallback


def infer_section(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip() or None
        break
    return None


def make_doc_id(source_path: str, fallback_name: str) -> str:
    base = source_path or fallback_name
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return slug or "document"


def build_document_record(path: Path, text: str, root_dir: Path | None = None) -> DocumentRecord:
    source_path = str(path.relative_to(root_dir)) if root_dir else str(path)
    source_name = path.name
    title = infer_title(text, fallback=path.stem)
    section = infer_section(text)
    return DocumentRecord(
        doc_id=make_doc_id(source_path=source_path, fallback_name=path.stem),
        source_path=source_path,
        source_name=source_name,
        title=title,
        text=text,
        section=section,
    )


def build_chunk_records(
    document: DocumentRecord,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    spans = chunk_document_with_spans(
        document.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        default_section=document.section,
    )
    records: list[ChunkRecord] = []

    for idx, span in enumerate(spans):
        chunk_id = f"{document.doc_id}__{idx:04d}"
        records.append(
            ChunkRecord(
                doc_id=document.doc_id,
                source_path=document.source_path,
                source_name=document.source_name,
                title=document.title,
                chunk_id=chunk_id,
                text=span.text,
                start_char=span.start_char,
                end_char=span.end_char,
                section=span.section,
            )
        )

    linked_records: list[ChunkRecord] = []
    for idx, record in enumerate(records):
        prev_chunk_id = records[idx - 1].chunk_id if idx > 0 else None
        next_chunk_id = records[idx + 1].chunk_id if idx + 1 < len(records) else None
        linked_records.append(
            ChunkRecord(
                doc_id=record.doc_id,
                source_path=record.source_path,
                source_name=record.source_name,
                title=record.title,
                chunk_id=record.chunk_id,
                text=record.text,
                start_char=record.start_char,
                end_char=record.end_char,
                section=record.section,
                prev_chunk_id=prev_chunk_id,
                next_chunk_id=next_chunk_id,
            )
        )

    return linked_records


def chunk_record_from_payload(payload: dict) -> ChunkRecord:
    return ChunkRecord(
        doc_id=payload.get("doc_id", "legacy_document"),
        source_path=payload.get("source_path", ""),
        source_name=payload.get("source_name", ""),
        title=payload.get("title", ""),
        chunk_id=payload.get("chunk_id", payload.get("doc_id", "legacy_document")),
        text=payload["text"],
        start_char=payload.get("start_char", 0),
        end_char=payload.get("end_char", len(payload["text"])),
        section=payload.get("section"),
        prev_chunk_id=payload.get("prev_chunk_id"),
        next_chunk_id=payload.get("next_chunk_id"),
    )


def legacy_text_to_chunk_record(text: str, idx: int) -> ChunkRecord:
    doc_id = f"legacy_document_{idx:04d}"
    chunk_id = f"{doc_id}__0000"
    return ChunkRecord(
        doc_id=doc_id,
        source_path="",
        source_name="",
        title="",
        chunk_id=chunk_id,
        text=text,
        start_char=0,
        end_char=len(text),
        section=None,
        prev_chunk_id=None,
        next_chunk_id=None,
    )
