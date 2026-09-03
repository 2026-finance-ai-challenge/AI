from collections.abc import Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from pgvector import HalfVector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from k_market_ai.rag.application.ports import RagRepository
from k_market_ai.rag.domain.chunker import chunk_sections, normalize_text
from k_market_ai.rag.domain.models import (
    EmbeddedChunk,
    IndexJob,
    MetadataEmbeddingJob,
    SearchHit,
    SectionKind,
    SourceSection,
)
from k_market_ai.rag.domain.selected_translation import translated_selection_exists
from k_market_ai.rag.infrastructure.payload_codec import decode_sections

INDEX_JOB_TYPE = "DISCLOSURE_EMBEDDING"
METADATA_JOB_TYPE = "DISCLOSURE_METADATA_EMBEDDING"


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
                    JOIN service_stock_universe AS universe
                      ON universe.stock_code = security.stock_code
                    JOIN disclosure_document AS document
                      ON document.disclosure_id = disclosure.id
                     AND document.is_current = TRUE
                     AND document.payload_zstd IS NOT NULL
                    WHERE job.job_type = %s
                      AND security.active
                      AND security.common_stock
                      AND pg_database_size(current_database()) < 55834574848
                      AND job.available_at <= CURRENT_TIMESTAMP
                      AND (
                          job.status = 'PENDING'
                          OR (
                              job.status = 'PROCESSING'
                              AND job.locked_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes'
                          )
                      )
                    ORDER BY disclosure.filed_date DESC, job.business_key DESC,
                             job.attempts DESC, job.available_at DESC, job.created_at DESC
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
                SELECT document.id, document.version_no, document.payload_zstd
                FROM disclosure
                JOIN security ON security.id = disclosure.security_id
                JOIN service_stock_universe AS universe
                  ON universe.stock_code = security.stock_code
                JOIN disclosure_document AS document
                  ON document.disclosure_id = disclosure.id AND document.is_current = TRUE
                WHERE disclosure.receipt_number = %s
                ORDER BY document.created_at, document.source_filename
                """,
                (receipt_number,),
            )
            document_rows = await cursor.fetchall()
            sections: list[SourceSection] = []
            for document_id_value, version_value, payload in document_rows:
                document_id = UUID(str(document_id_value))
                version = int(version_value)
                if payload is not None:
                    sections.extend(decode_sections(payload, document_id, version))
                    continue
                legacy_cursor = await connection.execute(
                    """
                    SELECT id, ordinal, section_kind, heading, text_content
                    FROM disclosure_section
                    WHERE document_id = %s
                    ORDER BY ordinal
                    """,
                    (document_id,),
                )
                legacy_rows = await legacy_cursor.fetchall()
                sections.extend(
                    SourceSection(
                        id=UUID(str(row[0])),
                        document_id=document_id,
                        document_version=version,
                        ordinal=int(row[1]),
                        kind=SectionKind(str(row[2])),
                        heading=None if row[3] is None else str(row[3]),
                        text=str(row[4]),
                    )
                    for row in legacy_rows
                )
        return sections

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

            current_cursor = await connection.execute(
                "SELECT id FROM disclosure_document WHERE disclosure_id = %s AND is_current",
                (disclosure_id,),
            )
            current_ids = {UUID(str(row[0])) for row in await current_cursor.fetchall()}
            if {item.chunk.document_id for item in chunks} != current_ids:
                # 색인 도중 원문 버전이 바뀌면 새 버전의 대기 작업을 완료 처리하지 않는다.
                return

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
                    SET content = NULL, embedding = EXCLUDED.embedding, is_current = TRUE
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
                        None,
                        chunk.content_hash,
                        HalfVector(np.asarray(embedded.embedding, dtype=np.float32)),
                        embedding_model,
                        embedding_dimensions,
                        chunker_version,
                    ),
                )

            # 현재 공시만 검색하므로 이전 버전 청크는 원문 ZIP으로 대체한다.
            await connection.execute(
                "DELETE FROM disclosure_chunk WHERE disclosure_id = %s AND is_current = FALSE",
                (disclosure_id,),
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

    async def claim_metadata_embedding_job(
        self,
        worker_id: str,
    ) -> MetadataEmbeddingJob | None:
        async with self._pool.connection() as connection, connection.transaction():
            cursor = await connection.execute(
                """
                WITH candidate AS (
                    SELECT job.id
                    FROM ingestion_job AS job
                    JOIN disclosure ON disclosure.receipt_number = job.business_key
                    JOIN security ON security.id = disclosure.security_id
                    JOIN service_stock_universe AS universe
                      ON universe.stock_code = security.stock_code
                    WHERE job.job_type = %s
                      AND security.active AND security.common_stock
                      AND job.available_at <= CURRENT_TIMESTAMP
                      AND (
                          job.status = 'PENDING'
                          OR (
                              job.status = 'PROCESSING'
                              AND job.locked_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes'
                          )
                      )
                    ORDER BY job.attempts DESC, job.available_at, job.created_at
                    FOR UPDATE OF job SKIP LOCKED
                    LIMIT 1
                )
                UPDATE ingestion_job AS job
                SET status = 'PROCESSING', attempts = attempts + 1,
                    locked_at = CURRENT_TIMESTAMP, locked_by = %s,
                    updated_at = CURRENT_TIMESTAMP
                FROM candidate
                WHERE job.id = candidate.id
                RETURNING job.business_key, job.attempts
                """,
                (METADATA_JOB_TYPE, worker_id),
            )
            row = await cursor.fetchone()
        return (
            None
            if row is None
            else MetadataEmbeddingJob(receipt_number=str(row[0]), attempts=int(row[1]))
        )

    async def load_metadata_embedding_text(self, receipt_number: str) -> str:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT concat_ws(
                    E'\\n', issuer.name_ko, issuer.name_en, security.stock_code,
                    disclosure.filed_date::TEXT, disclosure.disclosure_type,
                    disclosure.title_ko
                )
                FROM disclosure
                JOIN issuer ON issuer.id = disclosure.issuer_id
                JOIN security ON security.id = disclosure.security_id
                JOIN service_stock_universe AS universe
                  ON universe.stock_code = security.stock_code
                WHERE disclosure.receipt_number = %s
                """,
                (receipt_number,),
            )
            row = await cursor.fetchone()
        if row is None:
            raise LookupError("Disclosure does not exist")
        return str(row[0])

    async def complete_metadata_embedding_job(
        self,
        receipt_number: str,
        embedding: Sequence[float],
        embedding_model: str,
        embedding_dimensions: int,
        source_hash: str,
    ) -> None:
        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO disclosure_document_embedding (
                    disclosure_id, embedding, embedding_model, embedding_dimensions,
                    source_hash, created_at, updated_at
                )
                SELECT id, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                FROM disclosure
                WHERE receipt_number = %s
                ON CONFLICT (disclosure_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dimensions = EXCLUDED.embedding_dimensions,
                    source_hash = EXCLUDED.source_hash,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    HalfVector(np.asarray(embedding, dtype=np.float32)),
                    embedding_model,
                    embedding_dimensions,
                    source_hash,
                    receipt_number,
                ),
            )
            await connection.execute(
                """
                UPDATE ingestion_job
                SET status = 'COMPLETED', locked_at = NULL, locked_by = NULL,
                    last_error_code = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE job_type = %s AND business_key = %s
                """,
                (METADATA_JOB_TYPE, receipt_number),
            )

    async def retry_metadata_embedding_job(
        self,
        receipt_number: str,
        error_code: str,
        delay: timedelta,
    ) -> None:
        await self._update_metadata_job(
            receipt_number,
            "PENDING",
            error_code,
            delay,
        )

    async def fail_metadata_embedding_job(
        self,
        receipt_number: str,
        error_code: str,
    ) -> None:
        await self._update_metadata_job(receipt_number, "FAILED", error_code, timedelta())

    async def _update_metadata_job(
        self,
        receipt_number: str,
        status: str,
        error_code: str,
        delay: timedelta,
    ) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                UPDATE ingestion_job
                SET status = %s,
                    available_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                    locked_at = NULL, locked_by = NULL, last_error_code = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_type = %s AND business_key = %s
                """,
                (
                    status,
                    delay.total_seconds(),
                    error_code[:100],
                    METADATA_JOB_TYPE,
                    receipt_number,
                ),
            )

    async def selected_text_exists(
        self,
        receipt_number: str,
        section_id: UUID,
        normalized_text: str,
        translation_source_hash: str | None = None,
    ) -> bool:
        sections = await self.load_current_sections(receipt_number)
        section = next((section for section in sections if section.id == section_id), None)
        if section is None or not normalized_text:
            return False
        if translation_source_hash is None:
            return normalized_text in normalize_text(section.text)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT source_text, result_payload
                FROM translation_memory
                WHERE content_kind = 'DISCLOSURE_SECTION' AND source_hash = %s
                  AND target_locale = 'en' AND translation_version = 'disclosure-section-v4'
                  AND status = 'READY'
                """,
                (translation_source_hash,),
            )
            row = await cursor.fetchone()
        return row is not None and translated_selection_exists(
            section, normalized_text, translation_source_hash, row[0], row[1]
        )

    async def search(
        self,
        receipt_number: str,
        embedding: Sequence[float],
        embedding_model: str,
        selected_section_id: UUID | None,
        limit: int,
    ) -> list[SearchHit]:
        query_vector = HalfVector(np.asarray(embedding, dtype=np.float32))
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                WITH scoped AS (
                    SELECT chunk.*, document.version_no
                    FROM disclosure
                    JOIN security ON security.id = disclosure.security_id
                    JOIN service_stock_universe AS universe
                      ON universe.stock_code = security.stock_code
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
                SELECT scoped.id, scoped.document_id, scoped.version_no, scoped.chunk_index,
                       scoped.section_ids, scoped.first_ordinal, scoped.last_ordinal,
                       scoped.heading, scoped.content,
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
            missing_document_ids = {UUID(str(row[1])) for row in rows if row[8] is None}
            reconstructed: dict[tuple[UUID, int], str] = {}
            for document_id in missing_document_ids:
                document_cursor = await connection.execute(
                    """
                    SELECT version_no, payload_zstd
                    FROM disclosure_document
                    WHERE id = %s
                    """,
                    (document_id,),
                )
                document_row = await document_cursor.fetchone()
                if document_row is None:
                    continue
                version = int(document_row[0])
                if document_row[1] is not None:
                    sections = decode_sections(document_row[1], document_id, version)
                else:
                    section_cursor = await connection.execute(
                        """
                        SELECT id, ordinal, section_kind, heading, text_content
                        FROM disclosure_section
                        WHERE document_id = %s
                        ORDER BY ordinal
                        """,
                        (document_id,),
                    )
                    section_rows = await section_cursor.fetchall()
                    sections = [
                        SourceSection(
                            id=UUID(str(row[0])),
                            document_id=document_id,
                            document_version=version,
                            ordinal=int(row[1]),
                            kind=SectionKind(str(row[2])),
                            heading=None if row[3] is None else str(row[3]),
                            text=str(row[4]),
                        )
                        for row in section_rows
                    ]
                for chunk in chunk_sections(sections):
                    reconstructed[(document_id, chunk.chunk_index)] = chunk.content
        return [
            SearchHit(
                chunk_id=UUID(str(row[0])),
                document_id=UUID(str(row[1])),
                document_version=int(row[2]),
                section_ids=tuple(UUID(str(value)) for value in row[4]),
                first_ordinal=int(row[5]),
                last_ordinal=int(row[6]),
                heading=None if row[7] is None else str(row[7]),
                content=(
                    str(row[8])
                    if row[8] is not None
                    else reconstructed.get((UUID(str(row[1])), int(row[3])), "")
                ),
                score=float(row[9]),
                selected_priority=int(row[10]),
            )
            for row in rows
            if row[8] is not None or reconstructed.get((UUID(str(row[1])), int(row[3])))
        ]
