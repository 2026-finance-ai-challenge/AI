from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from k_market_ai.market.foreign_ownership_quantity_model import (
    ForeignOwnershipQuantityModel,
    ForeignOwnershipQuantityModelUnavailableError,
    ForeignOwnershipQuantityPoint,
)


@dataclass(frozen=True)
class ForeignOwnershipHistoryPoint:
    base_date: date
    foreign_owned_quantity: int
    foreign_limit_quantity: int


@dataclass(frozen=True)
class ForeignOwnershipForecast:
    min_rate: float
    base_rate: float
    max_rate: float
    observation_count: int
    observation_window_days: int
    confidence: float
    model_version: str
    base_date: date
    calculated_at: datetime
    source: str


class ForeignOwnershipForecastService:
    def __init__(self, model_path: Path | None = None) -> None:
        resolved_path = model_path or (
            Path(__file__).resolve().parents[1]
            / "model_store"
            / "foreign_ownership_quantity_ml.joblib"
        )
        self._model = ForeignOwnershipQuantityModel(resolved_path)

    def predict(
        self,
        stock_code: str,
        foreign_owned_quantity: int,
        total_listed_quantity: int,
        foreign_limit_quantity: int,
        base_date: date,
        history: list[ForeignOwnershipHistoryPoint],
    ) -> ForeignOwnershipForecast:
        points_by_date = {
            point.base_date: ForeignOwnershipQuantityPoint(
                stock_code=stock_code,
                base_date=point.base_date,
                foreign_owned_quantity=point.foreign_owned_quantity,
                foreign_limit_quantity=point.foreign_limit_quantity,
            )
            for point in history
            if point.foreign_owned_quantity > 0 and point.foreign_limit_quantity > 0
        }
        points_by_date[base_date] = ForeignOwnershipQuantityPoint(
            stock_code=stock_code,
            base_date=base_date,
            foreign_owned_quantity=foreign_owned_quantity,
            foreign_limit_quantity=foreign_limit_quantity,
        )
        points = sorted(points_by_date.values(), key=lambda point: point.base_date)
        prediction = self._model.predict(stock_code, points)
        first_date = points[0].base_date
        return ForeignOwnershipForecast(
            min_rate=_ownership_rate(prediction.lower_quantity, total_listed_quantity),
            base_rate=_ownership_rate(prediction.predicted_quantity, total_listed_quantity),
            max_rate=_ownership_rate(prediction.upper_quantity, total_listed_quantity),
            observation_count=len(points),
            observation_window_days=max(0, (base_date - first_date).days),
            confidence=prediction.confidence_score,
            model_version=prediction.model_version,
            base_date=base_date,
            calculated_at=datetime.now(UTC),
            source=prediction.source,
        )


def _ownership_rate(quantity: int, total_listed_quantity: int) -> float:
    if total_listed_quantity <= 0:
        raise ForeignOwnershipQuantityModelUnavailableError(
            "total listed quantity must be positive"
        )
    return round(quantity * 100 / total_listed_quantity, 6)
