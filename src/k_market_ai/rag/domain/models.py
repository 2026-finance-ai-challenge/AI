from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID


class SectionKind(StrEnum):
    TITLE = "TITLE"
    TEXT = "TEXT"
    TABLE = "TABLE"


@dataclass(frozen=True, slots=True)
class FilingCandidate:
    receipt_number: str
    stock_code: str
    title: str
    filed_date: date
    detected_at: datetime


@dataclass(frozen=True, slots=True)
class FilingEvidence:
    filing: FilingCandidate
    content: str
    section_ids: tuple[UUID, ...]
    retrieval_method: str


@dataclass(frozen=True, slots=True)
class SourceSection:
    id: UUID
    document_id: UUID
    document_version: int
    ordinal: int
    kind: SectionKind
    heading: str | None
    text: str


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    document_id: UUID
    document_version: int
    chunk_index: int
    section_ids: tuple[UUID, ...]
    first_ordinal: int
    last_ordinal: int
    heading: str | None
    content: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: ChunkDraft
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class IndexJob:
    receipt_number: str
    attempts: int


@dataclass(frozen=True, slots=True)
class MetadataEmbeddingJob:
    receipt_number: str
    attempts: int


@dataclass(frozen=True, slots=True)
class SelectedContext:
    section_id: UUID
    text: str
    translation_source_hash: str | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    document_version: int
    section_ids: tuple[UUID, ...]
    first_ordinal: int
    last_ordinal: int
    heading: str | None
    content: str
    score: float
    selected_priority: int


@dataclass(frozen=True, slots=True)
class Citation:
    id: str
    chunk_id: UUID
    document_id: UUID
    document_version: int
    section_ids: tuple[UUID, ...]
    first_ordinal: int
    last_ordinal: int
    heading: str | None
    excerpt: str


@dataclass(frozen=True, slots=True)
class RagAnswer:
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: tuple[Citation, ...]
    model: str | None
    prompt_version: str


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    sufficient_evidence: bool
    citation_ids: tuple[str, ...]
    refusal_reason: str | None
    model: str
    answer_locale: Literal["en", "ko"] = "en"
