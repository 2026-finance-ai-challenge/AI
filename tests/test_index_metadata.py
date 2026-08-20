import asyncio
import hashlib
from collections.abc import Sequence
from datetime import timedelta

from k_market_ai.rag.application.index_metadata import IndexMetadataHandler
from k_market_ai.rag.domain.models import MetadataEmbeddingJob


def test_metadata_job_stores_document_embedding() -> None:
    repository = FakeMetadataRepository()

    processed = asyncio.run(IndexMetadataHandler(repository, FakeEmbedding()).process_next())

    assert processed is True
    assert repository.completed is True
    assert repository.source_hash == hashlib.sha256(repository.text.encode()).hexdigest()


class FakeEmbedding:
    model = "test-embedding"
    dimensions = 3

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        assert texts
        return [(0.1, 0.2, 0.3)]


class FakeMetadataRepository:
    def __init__(self) -> None:
        self.text = "Samsung Electronics\n005930\nAnnual report"
        self.completed = False
        self.source_hash: str | None = None

    async def claim_metadata_embedding_job(
        self,
        worker_id: str,
    ) -> MetadataEmbeddingJob | None:
        assert worker_id
        return MetadataEmbeddingJob("20260818800670", 1)

    async def load_metadata_embedding_text(self, receipt_number: str) -> str:
        assert receipt_number
        return self.text

    async def complete_metadata_embedding_job(
        self,
        receipt_number: str,
        embedding: Sequence[float],
        embedding_model: str,
        embedding_dimensions: int,
        source_hash: str,
    ) -> None:
        assert receipt_number
        assert embedding_model == "test-embedding"
        assert embedding_dimensions == 3
        assert len(embedding) == 3
        self.completed = True
        self.source_hash = source_hash

    async def retry_metadata_embedding_job(
        self,
        receipt_number: str,
        error_code: str,
        delay: timedelta,
    ) -> None:
        raise AssertionError("retry must not be called")

    async def fail_metadata_embedding_job(
        self,
        receipt_number: str,
        error_code: str,
    ) -> None:
        raise AssertionError("fail must not be called")
