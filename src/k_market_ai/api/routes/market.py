from datetime import date, datetime
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.errors import AppError
from k_market_ai.market.foreign_ownership_quantity_model import (
    ForeignOwnershipQuantityModelUnavailableError,
)
from k_market_ai.market.service import (
    ForeignOwnershipForecastService,
    ForeignOwnershipHistoryPoint,
)

router = APIRouter(prefix="/internal/v1/market", tags=["market-forecast"])


class ForeignOwnershipHistoryPointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_date: date
    foreign_owned_quantity: int = Field(gt=0)
    foreign_limit_quantity: int = Field(gt=0)


class ForeignOwnershipForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_code: str = Field(pattern=r"^\d{6}$")
    foreign_owned_quantity: int = Field(gt=0)
    total_listed_quantity: int = Field(gt=0)
    foreign_limit_quantity: int = Field(gt=0)
    base_date: date
    history: list[ForeignOwnershipHistoryPointRequest] = Field(
        min_length=20,
        max_length=120,
    )


class ForeignOwnershipForecastResponse(BaseModel):
    min_rate: float = Field(ge=0)
    base_rate: float = Field(ge=0)
    max_rate: float = Field(ge=0)
    observation_count: int = Field(ge=20)
    observation_window_days: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    model_version: str
    base_date: date
    calculated_at: datetime
    source: str


@router.post("/foreign-ownership/forecast", response_model=ForeignOwnershipForecastResponse)
def forecast_foreign_ownership(
    body: ForeignOwnershipForecastRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> ForeignOwnershipForecastResponse:
    try:
        result = _service().predict(
            stock_code=body.stock_code,
            foreign_owned_quantity=body.foreign_owned_quantity,
            total_listed_quantity=body.total_listed_quantity,
            foreign_limit_quantity=body.foreign_limit_quantity,
            base_date=body.base_date,
            history=[
                ForeignOwnershipHistoryPoint(
                    base_date=point.base_date,
                    foreign_owned_quantity=point.foreign_owned_quantity,
                    foreign_limit_quantity=point.foreign_limit_quantity,
                )
                for point in body.history
            ],
        )
    except ForeignOwnershipQuantityModelUnavailableError as exception:
        raise AppError(
            code="FOREIGN_OWNERSHIP_MODEL_UNAVAILABLE",
            message="The foreign ownership forecast is unavailable.",
            status_code=503,
        ) from exception
    return ForeignOwnershipForecastResponse(**result.__dict__)


@lru_cache(maxsize=1)
def _service() -> ForeignOwnershipForecastService:
    return ForeignOwnershipForecastService()
