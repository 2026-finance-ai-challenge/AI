import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from k_market_ai.rag.infrastructure.postgres_repository import PostgresRagRepository


def test_claims_newest_disclosure_before_historical_embedding_backlog() -> None:
    cursor = SimpleNamespace(fetchone=AsyncMock(return_value=None))
    connection = SimpleNamespace(execute=AsyncMock(return_value=cursor))

    @asynccontextmanager
    async def context():
        yield connection

    connection.transaction = context
    repository = PostgresRagRepository(SimpleNamespace(connection=context))

    assert asyncio.run(repository.claim_index_job("priority-test")) is None

    sql, _ = connection.execute.call_args.args
    assert "ORDER BY disclosure.filed_date DESC, job.business_key DESC" in sql
