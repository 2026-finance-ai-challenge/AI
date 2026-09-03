from collections.abc import Sequence
from datetime import timedelta
from typing import Protocol
from uuid import UUID

from k_market_ai.core.answer_language import AnswerLocale
from k_market_ai.rag.domain.models import (
    EmbeddedChunk,
    GeneratedAnswer,
    IndexJob,
    MetadataEmbeddingJob,
    SearchHit,
    SourceSection,
)


class EmbeddingPort(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class AnswerPort(Protocol):
    async def answer(
        self,
        question: str,
        contexts: Sequence[tuple[str, SearchHit]],
        answer_locale: AnswerLocale = "auto",
    ) -> GeneratedAnswer: ...


class RagRepository(Protocol):
    async def claim_index_job(self, worker_id: str) -> IndexJob | None: ...

    async def load_current_sections(self, receipt_number: str) -> list[SourceSection]: ...

    async def complete_index_job(
        self,
        receipt_number: str,
        chunks: Sequence[EmbeddedChunk],
        embedding_model: str,
        embedding_dimensions: int,
        chunker_version: str,
    ) -> None: ...

    async def retry_index_job(
        self,
        receipt_number: str,
        error_code: str,
        delay: timedelta,
    ) -> None: ...

    async def fail_index_job(self, receipt_number: str, error_code: str) -> None: ...

    async def claim_metadata_embedding_job(
        self,
        worker_id: str,
    ) -> MetadataEmbeddingJob | None: ...

    async def load_metadata_embedding_text(self, receipt_number: str) -> str: ...

    async def complete_metadata_embedding_job(
        self,
        receipt_number: str,
        embedding: Sequence[float],
        embedding_model: str,
        embedding_dimensions: int,
        source_hash: str,
    ) -> None: ...

    async def retry_metadata_embedding_job(
        self,
        receipt_number: str,
        error_code: str,
        delay: timedelta,
    ) -> None: ...

    async def fail_metadata_embedding_job(
        self,
        receipt_number: str,
        error_code: str,
    ) -> None: ...

    async def selected_text_exists(
        self,
        receipt_number: str,
        section_id: UUID,
        normalized_text: str,
        translation_source_hash: str | None = None,
    ) -> bool: ...

    async def search(
        self,
        receipt_number: str,
        embedding: Sequence[float],
        embedding_model: str,
        selected_section_id: UUID | None,
        limit: int,
    ) -> list[SearchHit]: ...
