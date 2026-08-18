import logging
from datetime import timedelta
from uuid import uuid4

from k_market_ai.rag.application.ports import EmbeddingPort, RagRepository
from k_market_ai.rag.domain.chunker import CHUNKER_VERSION, chunk_sections
from k_market_ai.rag.domain.errors import RagIntegrityError
from k_market_ai.rag.domain.models import EmbeddedChunk, IndexJob

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5


class IndexDisclosureHandler:
    def __init__(self, repository: RagRepository, embedding: EmbeddingPort) -> None:
        self._repository = repository
        self._embedding = embedding
        self._worker_id = str(uuid4())

    async def process_next(self) -> bool:
        job = await self._repository.claim_index_job(self._worker_id)
        if job is None:
            return False

        try:
            sections = await self._repository.load_current_sections(job.receipt_number)
            chunks = chunk_sections(sections)
            if not chunks:
                raise RagIntegrityError("No indexable disclosure content")
            vectors = await self._embedding.embed([chunk.content for chunk in chunks])
            if len(vectors) != len(chunks):
                raise RagIntegrityError("Embedding count mismatch")
            if any(len(vector) != self._embedding.dimensions for vector in vectors):
                raise RagIntegrityError("Embedding dimension mismatch")
            embedded = [
                EmbeddedChunk(chunk=chunk, embedding=vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            await self._repository.complete_index_job(
                job.receipt_number,
                embedded,
                self._embedding.model,
                self._embedding.dimensions,
                CHUNKER_VERSION,
            )
        except Exception as exception:
            await self._handle_failure(job, exception)
        return True

    async def _handle_failure(self, job: IndexJob, exception: Exception) -> None:
        error_code = type(exception).__name__[:100]
        if job.attempts >= MAX_ATTEMPTS:
            await self._repository.fail_index_job(job.receipt_number, error_code)
        else:
            await self._repository.retry_index_job(
                job.receipt_number,
                error_code,
                timedelta(minutes=5),
            )
        logger.warning(
            "공시 색인 실패: receipt_number=%s error_type=%s",
            job.receipt_number,
            error_code,
        )
