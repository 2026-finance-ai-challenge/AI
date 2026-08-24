from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from k_market_ai.agent.service import MarketAgentService
from k_market_ai.api.router import api_router
from k_market_ai.core.config import Settings, get_settings
from k_market_ai.core.exception_handlers import register_exception_handlers
from k_market_ai.core.request_id import RequestIdMiddleware
from k_market_ai.news.runtime import NewsRuntime
from k_market_ai.news.service import NewsIntelligenceService
from k_market_ai.peers.service import GlobalPeerService
from k_market_ai.rag.application.ask_disclosure import AskDisclosureHandler
from k_market_ai.rag.application.disclosure_insight import DisclosureInsightService
from k_market_ai.rag.infrastructure.runtime import ApiRagRuntime
from k_market_ai.tax.service import TaxDocumentService
from k_market_ai.translations.service import TranslationService


def create_app(
    settings: Settings | None = None,
    rag_handler: AskDisclosureHandler | None = None,
    news_service: NewsIntelligenceService | None = None,
    disclosure_insight_service: DisclosureInsightService | None = None,
    agent_service: MarketAgentService | None = None,
    tax_document_service: TaxDocumentService | None = None,
    global_peer_service: GlobalPeerService | None = None,
    translation_service: TranslationService | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    docs_url = "/docs" if app_settings.docs_enabled else None
    openapi_url = "/openapi.json" if app_settings.docs_enabled else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime: ApiRagRuntime | None = None
        news_runtime: NewsRuntime | None = None
        if rag_handler is None and app_settings.api_rag_configured:
            runtime = ApiRagRuntime.create(app_settings)
            await runtime.open()
            app.state.rag_handler = runtime.handler
        if app_settings.openai_configured:
            news_runtime = NewsRuntime.create(app_settings)
            if news_service is None and news_runtime.service is not None:
                app.state.news_service = news_runtime.service
            if disclosure_insight_service is None:
                app.state.disclosure_insight_service = DisclosureInsightService(
                    news_runtime.openai,
                    app_settings,
                )
            if agent_service is None:
                app.state.agent_service = MarketAgentService(news_runtime.openai, app_settings)
            if tax_document_service is None:
                app.state.tax_document_service = TaxDocumentService(
                    news_runtime.openai, app_settings
                )
            if global_peer_service is None:
                app.state.global_peer_service = GlobalPeerService(news_runtime.openai, app_settings)
            if translation_service is None:
                app.state.translation_service = news_runtime.translation_service
        try:
            yield
        finally:
            if news_runtime is not None:
                await news_runtime.close()
            if runtime is not None:
                await runtime.close()

    app = FastAPI(
        title=app_settings.app_name,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=app_settings.allowed_hosts)
    app.add_middleware(RequestIdMiddleware)
    app.state.settings = app_settings
    app.state.rag_handler = rag_handler
    app.state.news_service = news_service
    app.state.disclosure_insight_service = disclosure_insight_service
    app.state.agent_service = agent_service
    app.state.tax_document_service = tax_document_service
    app.state.global_peer_service = global_peer_service
    app.state.translation_service = translation_service
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
