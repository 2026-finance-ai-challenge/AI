from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from k_market_ai.api.router import api_router
from k_market_ai.core.config import Settings, get_settings
from k_market_ai.core.exception_handlers import register_exception_handlers
from k_market_ai.core.request_id import RequestIdMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    docs_url = "/docs" if app_settings.docs_enabled else None
    openapi_url = "/openapi.json" if app_settings.docs_enabled else None

    app = FastAPI(
        title=app_settings.app_name,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=app_settings.allowed_hosts)
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
