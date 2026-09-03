import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from k_market_ai.rag.infrastructure.runtime import ApiRagRuntime


def test_rag_startup_requires_embedding_model_warmup():
    pool = SimpleNamespace(open=AsyncMock(), close=AsyncMock())
    client = SimpleNamespace(close=AsyncMock())
    embedding = SimpleNamespace(warmup=AsyncMock())
    runtime = ApiRagRuntime(pool, client, SimpleNamespace(), embedding)
    asyncio.run(runtime.open())
    pool.open.assert_awaited_once_with(wait=True)
    embedding.warmup.assert_awaited_once()
    pool.close.assert_not_awaited()


def test_rag_startup_closes_resources_when_model_cache_is_unwritable():
    pool = SimpleNamespace(open=AsyncMock(), close=AsyncMock())
    client = SimpleNamespace(close=AsyncMock())
    embedding = SimpleNamespace(warmup=AsyncMock(side_effect=PermissionError("model cache")))
    runtime = ApiRagRuntime(pool, client, SimpleNamespace(), embedding)
    with pytest.raises(PermissionError):
        asyncio.run(runtime.open())
    pool.close.assert_awaited_once()
    client.close.assert_awaited_once()
