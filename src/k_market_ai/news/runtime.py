from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from k_market_ai.core.config import Settings
from k_market_ai.news.classifier import HanaNewsSignalClassifier
from k_market_ai.news.service import NewsIntelligenceService


@dataclass(slots=True)
class NewsRuntime:
    openai: AsyncOpenAI
    service: NewsIntelligenceService | None

    @classmethod
    def create(cls, settings: Settings) -> NewsRuntime:
        if settings.openai_api_key is None:
            raise RuntimeError("News AI configuration is incomplete")
        client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=30.0,
            max_retries=2,
        )
        service = None
        if settings.hana_project_root is not None:
            classifier = HanaNewsSignalClassifier(
                settings.hana_project_root,
                expected_commit=settings.hana_expected_commit,
                runtime_environment=settings.environment,
            )
            service = NewsIntelligenceService(client, settings, classifier)
        return cls(openai=client, service=service)

    async def close(self) -> None:
        await self.openai.close()
