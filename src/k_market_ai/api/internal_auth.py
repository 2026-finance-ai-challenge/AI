import secrets
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError

bearer = HTTPBearer(auto_error=False)


async def authenticate_internal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    settings: Settings = request.app.state.settings
    expected = settings.service_token
    if expected is None:
        raise AppError(
            code="SERVICE_NOT_CONFIGURED",
            message="The internal AI service is not configured.",
            status_code=503,
        )
    if credentials is None or not secrets.compare_digest(
        credentials.credentials,
        expected.get_secret_value(),
    ):
        raise AppError(code="UNAUTHORIZED", message="Authentication is required.", status_code=401)
