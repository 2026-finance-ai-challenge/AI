import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from k_market_ai.rag.infrastructure.postgres_repository import PostgresRagRepository


def test_stale_index_does_not_complete_reparsed_document_job() -> None:
    current_id, old_id, disclosure_id = uuid4(), uuid4(), uuid4()
    connection = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(fetchone=AsyncMock(return_value=(disclosure_id,))),
                SimpleNamespace(fetchall=AsyncMock(return_value=[(current_id,)])),
            ]
        )
    )

    @asynccontextmanager
    async def context():
        yield connection

    connection.transaction = context
    pool = SimpleNamespace(connection=context)
    repository = PostgresRagRepository(pool)
    chunks = [SimpleNamespace(chunk=SimpleNamespace(document_id=old_id))]
    asyncio.run(repository.complete_index_job("20260902000001", chunks, "test", 384, "v1"))
    assert connection.execute.await_count == 2
    assert all("SELECT" in call.args[0] for call in connection.execute.await_args_list)
