from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from k_market_ai.core.config import Settings
from k_market_ai.rag.application.ask_disclosure import AskDisclosureHandler
from k_market_ai.rag.application.index_disclosure import IndexDisclosureHandler
from k_market_ai.rag.infrastructure.local_embedding import LocalEmbeddingAdapter
from k_market_ai.rag.infrastructure.openai_answer import OpenAIAnswerAdapter
from k_market_ai.rag.infrastructure.postgres_repository import (
    PostgresRagRepository,
    create_pool,
)


@dataclass(slots=True)
class ApiRagRuntime:
    pool: AsyncConnectionPool[AsyncConnection[object]]
    openai: AsyncOpenAI
    handler: AskDisclosureHandler

    @classmethod
    def create(cls, settings: Settings) -> ApiRagRuntime:
        if settings.database_url is None or settings.openai_api_key is None:
            raise RuntimeError("RAG API configuration is incomplete")
        pool = create_pool(settings.database_url.get_secret_value())
        repository = PostgresRagRepository(pool)
        embedding = LocalEmbeddingAdapter()
        client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=30.0,
            max_retries=2,
        )
        return cls(
            pool=pool,
            openai=client,
            handler=AskDisclosureHandler(
                repository,
                embedding,
                OpenAIAnswerAdapter(client),
            ),
        )

    async def open(self) -> None:
        await self.pool.open(wait=True)

    async def close(self) -> None:
        await self.pool.close()
        await self.openai.close()


@dataclass(slots=True)
class WorkerRagRuntime:
    pool: AsyncConnectionPool[AsyncConnection[object]]
    handler: IndexDisclosureHandler

    @classmethod
    def create(cls, settings: Settings) -> WorkerRagRuntime:
        if settings.database_url is None:
            raise RuntimeError("RAG worker database configuration is incomplete")
        pool = create_pool(settings.database_url.get_secret_value())
        repository = PostgresRagRepository(pool)
        return cls(
            pool=pool,
            handler=IndexDisclosureHandler(repository, LocalEmbeddingAdapter()),
        )

    async def open(self) -> None:
        await self.pool.open(wait=True)

    async def close(self) -> None:
        await self.pool.close()
