from collections.abc import Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from k_market_ai.rag.application.ports import RagRepository
from k_market_ai.rag.domain.chunker import normalize_text
from k_market_ai.rag.domain.models import (
    EmbeddedChunk,
    IndexJob,
    SearchHit,
    SectionKind,
    SourceSection,
)

INDEX_JOB_TYPE = "DISCLOSURE_EMBEDDING"


async def configure_vector(connection: AsyncConnection[Any]) -> None:
    await register_vector_async(connection)


def create_pool(database_url: str) -> AsyncConnectionPool[AsyncConnection[Any]]:
    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        timeout=5,
        open=False,
        configure=configure_vector,
    )


class PostgresRagRepository(RagRepository):
    def __init__(self, pool: AsyncConnectionPool[AsyncConnection[Any]]) -> None:
        self._pool = pool

    async def claim_index_job(self, worker_id: str) -> IndexJob | None:
        async with self._pool.connection() as connection, connection.transaction():
            cursor = await connection.execute(
                """
                WITH candidate AS (
                    SELECT job.id
                    FROM ingestion_job AS job
                    JOIN disclosure ON disclosure.receipt_number = job.business_key
                    JOIN security ON security.id = disclosure.security_id
                    WHERE job.job_type = %s
                      AND security.active
                      AND security.common_stock
                      AND job.available_at <= CURRENT_TIMESTAMP
                      AND (
                          job.status = 'PENDING'
                          OR (
                              job.status = 'PROCESSING'
                              AND job.locked_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes'
                          )
                      )
                    ORDER BY (job.attempts > 0) DESC, job.available_at, job.created_at
                    FOR UPDATE OF job SKIP LOCKED
                    LIMIT 1
                )
                UPDATE ingestion_job AS job
                SET status = 'PROCESSING',
                    attempts = attempts + 1,
                    locked_at = CURRENT_TIMESTAMP,
                    locked_by = %s,
                    updated_at = CURRENT_TIMESTAMP
                FROM candidate
                WHERE job.id = candidate.id
                RETURNING job.business_key, job.attempts
                """,
                (INDEX_JOB_TYPE, worker_id),
            )
            row = await cursor.fetchone()
        return None if row is None else IndexJob(receipt_number=str(row[0]), attempts=int(row[1]))

    async def load_current_sections(self, receipt_number: str) -> list[SourceSection]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT section.id, document.id, document.version_no, section.ordinal,
                       section.section_kind, section.heading, section.text_content
                FROM disclosure
                JOIN disclosure_document AS document
                  ON document.disclosure_id = disclosure.id AND document.is_current = TRUE
                JOIN disclosure_section AS section ON section.document_id = document.id
                WHERE disclosure.receipt_number = %s
                ORDER BY document.created_at, document.source_filename, section.ordinal
                """,
                (receipt_number,),
            )
            rows = await cursor.fetchall()
        return [
            SourceSection(
                id=UUID(str(row[0])),
                document_id=UUID(str(row[1])),
                document_version=int(row[2]),
                ordinal=int(row[3]),
                kind=SectionKind(str(row[4])),
                heading=None if row[5] is None else str(row[5]),
                text=str(row[6]),
            )
            for row in rows
        ]

    async def complete_index_job(
        self,
        receipt_number: str,
        chunks: Sequence[EmbeddedChunk],
        embedding_model: str,
        embedding_dimensions: int,
        chunker_version: str,
    ) -> None:
        async with self._pool.connection() as connection, connection.transaction():
            disclosure_cursor = await connection.execute(
                "SELECT id FROM disclosure WHERE receipt_number = %s FOR UPDATE",
                (receipt_number,),
            )
            disclosure_row = await disclosure_cursor.fetchone()
            if disclosure_row is None:
                raise LookupError("Disclosure does not exist")
            disclosure_id = UUID(str(disclosure_row[0]))

            await connection.execute(
                "UPDATE disclosure_chunk SET is_current = FALSE WHERE disclosure_id = %s",
                (disclosure_id,),
            )
            for embedded in chunks:
                chunk = embedded.chunk
                await connection.execute(
                    """
                    INSERT INTO disclosure_chunk (
                        id, disclosure_id, document_id, chunk_index, section_ids,
                        first_ordinal, last_ordinal, heading, content, content_hash,
                        embedding, embedding_model, embedding_dimensions,
                        chunker_version, is_current, created_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (
                        document_id, chunk_index, content_hash, embedding_model, chunker_version
                    ) DO UPDATE
                    SET embedding = EXCLUDED.embedding, is_current = TRUE
                    """,
                    (
                        uuid4(),
                        disclosure_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        list(chunk.section_ids),
                        chunk.first_ordinal,
                        chunk.last_ordinal,
                        chunk.heading,
                        chunk.content,
                        chunk.content_hash,
                        list(embedded.embedding),
                        embedding_model,
                        embedding_dimensions,
                        chunker_version,
                    ),
                )

            await connection.execute(
                """
                UPDATE disclosure
                SET index_status = 'READY', updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (disclosure_id,),
            )
            await connection.execute(
                """
                UPDATE ingestion_job
                SET status = 'COMPLETED', locked_at = NULL, locked_by = NULL,
                    last_error_code = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE job_type = %s AND business_key = %s
                """,
                (INDEX_JOB_TYPE, receipt_number),
            )

    async def retry_index_job(
        self,
        receipt_number: str,
        error_code: str,
        delay: timedelta,
    ) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                UPDATE ingestion_job
                SET status = 'PENDING',
                    available_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                    locked_at = NULL, locked_by = NULL, last_error_code = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_type = %s AND business_key = %s
                """,
                (delay.total_seconds(), error_code[:100], INDEX_JOB_TYPE, receipt_number),
            )

    async def fail_index_job(self, receipt_number: str, error_code: str) -> None:
        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute(
                """
                UPDATE ingestion_job
                SET status = 'FAILED', locked_at = NULL, locked_by = NULL,
                    last_error_code = %s, updated_at = CURRENT_TIMESTAMP
                WHERE job_type = %s AND business_key = %s
                """,
                (error_code[:100], INDEX_JOB_TYPE, receipt_number),
            )
            await connection.execute(
                """
                UPDATE disclosure
                SET index_status = 'FAILED', updated_at = CURRENT_TIMESTAMP
                WHERE receipt_number = %s
                """,
                (receipt_number,),
            )

    async def selected_text_exists(
        self,
        receipt_number: str,
        section_id: UUID,
        normalized_text: str,
    ) -> bool:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT section.text_content
                FROM disclosure
                JOIN disclosure_document AS document
                  ON document.disclosure_id = disclosure.id AND document.is_current = TRUE
                JOIN disclosure_section AS section ON section.document_id = document.id
                WHERE disclosure.receipt_number = %s AND section.id = %s
                """,
                (receipt_number, section_id),
            )
            row = await cursor.fetchone()
        return row is not None and normalized_text in normalize_text(str(row[0]))

    async def search(
        self,
        receipt_number: str,
        embedding: Sequence[float],
        embedding_model: str,
        selected_section_id: UUID | None,
        limit: int,
    ) -> list[SearchHit]:
        query_vector = np.asarray(embedding, dtype=np.float32)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                WITH scoped AS (
                    SELECT chunk.*, document.version_no
                    FROM disclosure
                    JOIN disclosure_document AS document
                      ON document.disclosure_id = disclosure.id AND document.is_current = TRUE
                    JOIN disclosure_chunk AS chunk
                      ON chunk.document_id = document.id AND chunk.is_current = TRUE
                    WHERE disclosure.receipt_number = %s
                      AND disclosure.index_status = 'READY'
                      AND chunk.embedding_model = %s
                ),
                selected_chunk AS (
                    SELECT document_id, chunk_index
                    FROM scoped
                    WHERE %s::UUID IS NOT NULL AND %s::UUID = ANY(section_ids)
                    ORDER BY chunk_index
                    LIMIT 1
                )
                SELECT scoped.id, scoped.document_id, scoped.version_no, scoped.section_ids,
                       scoped.first_ordinal, scoped.last_ordinal, scoped.heading, scoped.content,
                       1 - (scoped.embedding <=> %s) AS score,
                       CASE
                           WHEN %s::UUID IS NOT NULL AND %s::UUID = ANY(scoped.section_ids) THEN 0
                           WHEN selected_chunk.document_id = scoped.document_id
                                AND abs(selected_chunk.chunk_index - scoped.chunk_index) = 1 THEN 1
                           ELSE 2
                       END AS selected_priority
                FROM scoped
                LEFT JOIN selected_chunk ON TRUE
                ORDER BY selected_priority, scoped.embedding <=> %s
                LIMIT %s
                """,
                (
                    receipt_number,
                    embedding_model,
                    selected_section_id,
                    selected_section_id,
                    query_vector,
                    selected_section_id,
                    selected_section_id,
                    query_vector,
                    limit,
                ),
            )
            rows = await cursor.fetchall()
        return [
            SearchHit(
                chunk_id=UUID(str(row[0])),
                document_id=UUID(str(row[1])),
                document_version=int(row[2]),
                section_ids=tuple(UUID(str(value)) for value in row[3]),
                first_ordinal=int(row[4]),
                last_ordinal=int(row[5]),
                heading=None if row[6] is None else str(row[6]),
                content=str(row[7]),
                score=float(row[8]),
                selected_priority=int(row[9]),
            )
            for row in rows
        ]
