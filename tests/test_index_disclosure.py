import asyncio
from collections.abc import Sequence
from datetime import timedelta
from uuid import uuid4

from k_market_ai.rag.application.index_disclosure import IndexDisclosureHandler
from k_market_ai.rag.domain.models import EmbeddedChunk, IndexJob, SectionKind, SourceSection


def test_index_job_stores_versioned_chunks() -> None:
    document_id = uuid4()
    repository = FakeIndexRepository(
        job=IndexJob(receipt_number="20260818800670", attempts=1),
        sections=[
            SourceSection(
                id=uuid4(),
                document_id=document_id,
                document_version=2,
                ordinal=0,
                kind=SectionKind.TEXT,
                heading="Revenue",
                text="Revenue increased due to overseas demand.",
            )
        ],
    )

    processed = asyncio.run(IndexDisclosureHandler(repository, FakeEmbedding()).process_next())

    assert processed is True
    assert repository.completed is not None
    assert repository.completed[0].chunk.document_version == 2
    assert repository.embedding_model == "test-embedding"
    assert repository.embedding_dimensions == 3


def test_final_index_failure_marks_job_failed() -> None:
    repository = FakeIndexRepository(
        job=IndexJob(receipt_number="20260818800670", attempts=5),
        sections=[],
    )

    processed = asyncio.run(IndexDisclosureHandler(repository, FakeEmbedding()).process_next())

    assert processed is True
    assert repository.failed is True
    assert repository.retried is False


class FakeEmbedding:
    model = "test-embedding"
    dimensions = 3

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(0.1, 0.2, 0.3) for _ in texts]


class FakeIndexRepository:
    def __init__(self, job: IndexJob, sections: list[SourceSection]) -> None:
        self._job = job
        self._sections = sections
        self.completed: Sequence[EmbeddedChunk] | None = None
        self.embedding_model: str | None = None
        self.embedding_dimensions: int | None = None
        self.failed = False
        self.retried = False

    async def claim_index_job(self, worker_id: str) -> IndexJob | None:
        assert worker_id
        return self._job

    async def load_current_sections(self, receipt_number: str) -> list[SourceSection]:
        return self._sections

    async def complete_index_job(
        self,
        receipt_number: str,
        chunks: Sequence[EmbeddedChunk],
        embedding_model: str,
        embedding_dimensions: int,
        chunker_version: str,
    ) -> None:
        assert chunker_version
        self.completed = chunks
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

    async def retry_index_job(
        self,
        receipt_number: str,
        error_code: str,
        delay: timedelta,
    ) -> None:
        self.retried = True

    async def fail_index_job(self, receipt_number: str, error_code: str) -> None:
        self.failed = True
