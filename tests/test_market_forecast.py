import secrets
from datetime import date, timedelta

from fastapi.testclient import TestClient

from k_market_ai.core.config import Settings
from k_market_ai.main import create_app


def test_foreign_ownership_forecast_uses_promoted_model() -> None:
    token = secrets.token_urlsafe(24)
    app = create_app(
        Settings(environment="test", allowed_hosts=["testserver"], service_token=token)
    )
    start = date(2026, 1, 2)
    history = [
        {
            "base_date": (start + timedelta(days=index)).isoformat(),
            "foreign_owned_quantity": 87_000_000 + index * 10_000,
            "foreign_limit_quantity": 184_065_552,
        }
        for index in range(30)
    ]

    with TestClient(app) as client:
        missing_auth = client.post(
            "/internal/v1/market/foreign-ownership/forecast",
            json={
                "stock_code": "003490",
                "foreign_owned_quantity": 87_290_000,
                "total_listed_quantity": 368_220_661,
                "foreign_limit_quantity": 184_065_552,
                "base_date": history[-1]["base_date"],
                "history": history,
            },
        )
        response = client.post(
            "/internal/v1/market/foreign-ownership/forecast",
            headers={"authorization": f"Bearer {token}"},
            json={
                "stock_code": "003490",
                "foreign_owned_quantity": 87_290_000,
                "total_listed_quantity": 368_220_661,
                "foreign_limit_quantity": 184_065_552,
                "base_date": history[-1]["base_date"],
                "history": history,
            },
        )

    assert missing_auth.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["observation_count"] == 30
    assert payload["model_version"] == "kmarket-foreign-owned-quantity-ml-v2"
    assert payload["source"].startswith("KMARKET_AI_")
    assert payload["min_rate"] <= payload["base_rate"] <= payload["max_rate"]
