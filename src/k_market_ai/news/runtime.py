from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from k_market_ai.core.config import Settings
from k_market_ai.news.service import NewsIntelligenceService


@dataclass(slots=True)
class NewsRuntime:
    openai: AsyncOpenAI
    service: NewsIntelligenceService

    @classmethod
    def create(cls, settings: Settings) -> NewsRuntime:
        if settings.openai_api_key is None:
            raise RuntimeError("News AI configuration is incomplete")
        client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=30.0,
            max_retries=2,
        )
        return cls(openai=client, service=NewsIntelligenceService(client, settings))

    async def close(self) -> None:
        await self.openai.close()
