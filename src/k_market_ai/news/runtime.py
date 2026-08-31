from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from k_market_ai.core.config import Settings
from k_market_ai.news.classifier import FinancialSignalClassifier
from k_market_ai.news.service import NewsIntelligenceService
from k_market_ai.translations.service import TranslationService


@dataclass(slots=True)
class NewsRuntime:
    openai: AsyncOpenAI
    service: NewsIntelligenceService | None
    translation_service: TranslationService

    @classmethod
    def create(cls, settings: Settings) -> NewsRuntime:
        if settings.openai_api_key is None:
            raise RuntimeError("News AI configuration is incomplete")
        client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=30.0,
            # 작업 큐가 지수 백오프를 담당하므로 SDK 중첩 재시도를 막는다.
            max_retries=0,
        )
        service = None
        if settings.model_bundle_root is not None:
            classifier = FinancialSignalClassifier(
                settings.model_bundle_root,
                expected_commit=settings.model_bundle_commit,
                runtime_environment=settings.environment,
            )
            service = NewsIntelligenceService(client, settings, classifier)
        return cls(
            openai=client,
            service=service,
            translation_service=TranslationService(client, settings),
        )

    async def close(self) -> None:
        await self.openai.close()
