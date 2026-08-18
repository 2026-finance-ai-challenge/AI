import hashlib
import re
from collections.abc import Sequence

from k_market_ai.rag.domain.models import ChunkDraft, SourceSection

CHUNKER_VERSION = "section-window-v1"
TARGET_CHARS = 2_400
MAX_CHARS = 3_600
OVERLAP_CHARS = 300


def chunk_sections(sections: Sequence[SourceSection]) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    by_document: dict[object, list[SourceSection]] = {}
    for section in sections:
        by_document.setdefault(section.document_id, []).append(section)

    for document_sections in by_document.values():
        ordered = sorted(document_sections, key=lambda section: section.ordinal)
        chunks.extend(_chunk_document(ordered))
    return chunks


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _chunk_document(sections: Sequence[SourceSection]) -> list[ChunkDraft]:
    if not sections:
        return []

    result: list[ChunkDraft] = []
    pending: list[SourceSection] = []
    pending_parts: list[str] = []

    def flush() -> None:
        if not pending:
            return
        result.append(_draft(pending, "\n\n".join(pending_parts), len(result)))
        pending.clear()
        pending_parts.clear()

    for section in sections:
        text = _section_text(section)
        if not text:
            continue
        if len(text) > MAX_CHARS:
            flush()
            for window in _windows(text):
                result.append(_draft([section], window, len(result)))
            continue

        projected = sum(len(part) for part in pending_parts) + len(text) + 2 * len(pending_parts)
        if pending and (projected > MAX_CHARS or sum(map(len, pending_parts)) >= TARGET_CHARS):
            flush()
        pending.append(section)
        pending_parts.append(text)
    flush()
    return result


def _section_text(section: SourceSection) -> str:
    text = normalize_text(section.text)
    heading = normalize_text(section.heading or "")
    if not text:
        return ""
    if heading and heading != text:
        return f"{heading}\n{text}"
    return text


def _windows(text: str) -> list[str]:
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        windows.append(text[start:end])
        if end == len(text):
            break
        start = end - OVERLAP_CHARS
    return windows


def _draft(sections: Sequence[SourceSection], content: str, chunk_index: int) -> ChunkDraft:
    first = sections[0]
    normalized = content.strip()
    return ChunkDraft(
        document_id=first.document_id,
        document_version=first.document_version,
        chunk_index=chunk_index,
        section_ids=tuple(dict.fromkeys(section.id for section in sections)),
        first_ordinal=min(section.ordinal for section in sections),
        last_ordinal=max(section.ordinal for section in sections),
        heading=next((section.heading for section in sections if section.heading), None),
        content=normalized,
        content_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )
