import asyncio

import pytest

from k_market_ai.api.routes import translations
from k_market_ai.core.errors import AppError


def test_deadline_cancels_provider_and_includes_queue_wait(monkeypatch):
    monkeypatch.setattr(translations, "GENERATION_DEADLINE_SECONDS", 0.01)
    cancelled = []

    async def pending():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    async def run():
        with pytest.raises(AppError) as error:
            async with translations.generation_deadline():
                await pending()
        assert error.value.code == "AI_PROVIDER_TIMEOUT"

    asyncio.run(run())
    assert cancelled == [True]
