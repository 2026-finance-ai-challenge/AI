import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from k_market_ai.core.errors import AppError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)


async def app_error_handler(request: Request, exception: Exception) -> JSONResponse:
    error = _as_app_error(exception)
    return JSONResponse(
        status_code=error.status_code,
        content=_error_body(request, error.code, error.message),
    )


async def validation_error_handler(request: Request, exception: Exception) -> JSONResponse:
    validation_error = _as_validation_error(exception)
    violations = [
        {
            "location": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in validation_error.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            **_error_body(request, "INVALID_REQUEST", "The request is invalid."),
            "violations": violations,
        },
    )


async def unexpected_error_handler(request: Request, exception: Exception) -> JSONResponse:
    logger.error("처리하지 못한 서버 오류: %s", type(exception).__name__)
    return JSONResponse(
        status_code=500,
        content=_error_body(
            request,
            "INTERNAL_SERVER_ERROR",
            "An unexpected error occurred.",
        ),
    )


def _error_body(request: Request, code: str, message: str) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "request_id": getattr(request.state, "request_id", None),
    }


def _as_app_error(exception: Exception) -> AppError:
    if not isinstance(exception, AppError):
        raise TypeError from exception
    return exception


def _as_validation_error(exception: Exception) -> RequestValidationError:
    if not isinstance(exception, RequestValidationError):
        raise TypeError from exception
    return exception
