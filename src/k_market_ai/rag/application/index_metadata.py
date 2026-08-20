import hashlib
import logging
from datetime import timedelta
from uuid import uuid4

from k_market_ai.rag.application.ports import EmbeddingPort, RagRepository
from k_market_ai.rag.domain.errors import RagIntegrityError
from k_market_ai.rag.domain.models import MetadataEmbeddingJob

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5


class IndexMetadataHandler:
    def __init__(self, repository: RagRepository, embedding: EmbeddingPort) -> None:
        self._repository = repository
        self._embedding = embedding
        self._worker_id = str(uuid4())

    async def process_next(self) -> bool:
        job = await self._repository.claim_metadata_embedding_job(self._worker_id)
        if job is None:
            return False
        try:
            text = await self._repository.load_metadata_embedding_text(job.receipt_number)
            if not text.strip():
                raise RagIntegrityError("No metadata embedding content")
            vectors = await self._embedding.embed([text])
            if len(vectors) != 1 or len(vectors[0]) != self._embedding.dimensions:
                raise RagIntegrityError("Metadata embedding dimension mismatch")
            source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            await self._repository.complete_metadata_embedding_job(
                job.receipt_number,
                vectors[0],
                self._embedding.model,
                self._embedding.dimensions,
                source_hash,
            )
        except Exception as exception:
            await self._handle_failure(job, exception)
        return True

    async def _handle_failure(
        self,
        job: MetadataEmbeddingJob,
        exception: Exception,
    ) -> None:
        error_code = type(exception).__name__[:100]
        if job.attempts >= MAX_ATTEMPTS:
            await self._repository.fail_metadata_embedding_job(job.receipt_number, error_code)
        else:
            await self._repository.retry_metadata_embedding_job(
                job.receipt_number,
                error_code,
                timedelta(minutes=5),
            )
        logger.warning(
            "공시 메타데이터 색인 실패: receipt_number=%s error_type=%s",
            job.receipt_number,
            error_code,
        )
