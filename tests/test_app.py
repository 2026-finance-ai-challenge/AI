from fastapi import FastAPI, Query
from fastapi.testclient import TestClient

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.main import create_app


def test_health() -> None:
    app = create_app(Settings(environment="test", allowed_hosts=["testserver"]))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_untrusted_host_is_rejected() -> None:
    app = create_app(Settings(environment="test", allowed_hosts=["testserver"]))

    with TestClient(app) as client:
        response = client.get("/health", headers={"host": "untrusted.example"})

    assert response.status_code == 400


def test_docs_are_disabled_in_production() -> None:
    app = create_app(Settings(environment="production", allowed_hosts=["testserver"]))

    with TestClient(app) as client:
        response = client.get("/docs")

    assert response.status_code == 404


def test_app_error_returns_safe_message() -> None:
    app = _test_app()

    @app.get("/test/app-error")
    async def raise_app_error() -> None:
        raise AppError(code="NOT_FOUND", message="The resource was not found.", status_code=404)

    with TestClient(app) as client:
        response = client.get("/test/app-error")

    assert response.status_code == 404
    assert response.json()["message"] == "The resource was not found."
    assert response.json()["request_id"]


def test_validation_error_does_not_echo_input() -> None:
    app = _test_app()

    @app.get("/test/validation")
    async def validate(limit: int = Query(ge=1, le=100)) -> dict[str, int]:
        return {"limit": limit}

    with TestClient(app) as client:
        response = client.get("/test/validation", params={"limit": "private-value"})

    body = response.json()
    assert response.status_code == 422
    assert body["code"] == "INVALID_REQUEST"
    assert "private-value" not in response.text


def test_unexpected_error_hides_internal_message() -> None:
    app = _test_app()

    @app.get("/test/unexpected")
    async def raise_unexpected_error() -> None:
        raise RuntimeError("database-password")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test/unexpected")

    assert response.status_code == 500
    assert response.json()["message"] == "An unexpected error occurred."
    assert "database-password" not in response.text


def _test_app() -> FastAPI:
    return create_app(Settings(environment="test", allowed_hosts=["testserver"]))
