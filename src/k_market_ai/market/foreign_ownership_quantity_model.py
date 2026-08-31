from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from math import exp, sqrt
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Protocol, cast

import joblib  # type: ignore[import-untyped]

MODEL_SCHEMA_VERSION = "foreign-ownership-owned-quantity/v2"
MODEL_VERSION = "kmarket-foreign-owned-quantity-ml-v2"
MIN_HISTORY_OBSERVATIONS = 20
RUNTIME_POLICY_ML = "ml"
RUNTIME_POLICY_BASELINE = "baseline"
RUNTIME_POLICY_MEAN_DELTA_20 = "mean_delta_20"
RUNTIME_POLICY_MEDIAN_MULTI_DELTA = "median_multi_delta"
RUNTIME_POLICY_STALE_GUARDED_MEAN_DELTA_20 = "stale_guarded_mean_delta_20"
RUNTIME_POLICY_MICRO_MEDIAN_DELTA_3 = "micro_median_delta_3"
PREDICTION_MODE_DELTA_RATIO = "delta_ratio"
PREDICTION_MODE_DELTA_QUANTITY = "delta_quantity"
PREDICTION_MODE_TARGET_QUANTITY = "target_quantity"
PREDICTION_MODE_LOG_DELTA_RATIO = "log_delta_ratio"
PREDICTION_MODE_RESIDUAL_PREFIX = "residual:"
RUNTIME_POLICIES: tuple[str, ...] = (
    RUNTIME_POLICY_BASELINE,
    RUNTIME_POLICY_STALE_GUARDED_MEAN_DELTA_20,
    RUNTIME_POLICY_MEDIAN_MULTI_DELTA,
    RUNTIME_POLICY_MEAN_DELTA_20,
    RUNTIME_POLICY_MICRO_MEDIAN_DELTA_3,
    RUNTIME_POLICY_ML,
)
FEATURE_NAMES: tuple[str, ...] = (
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_10",
    "lag_20",
    "delta_1",
    "delta_3",
    "delta_5",
    "delta_10",
    "delta_20",
    "pct_delta_1",
    "pct_delta_3",
    "pct_delta_5",
    "pct_delta_10",
    "pct_delta_20",
    "mean_3",
    "mean_5",
    "mean_10",
    "mean_20",
    "std_5",
    "std_20",
    "min_20",
    "max_20",
    "range_20",
    "latest_month",
    "latest_day_of_month",
    "latest_day_of_week",
    "latest_ordinal_mod_7",
    "latest_ordinal_mod_30",
    "observations_since_last_change",
    "days_since_last_change",
    "prior_change_count",
    "prior_change_rate",
    "mean_days_between_changes",
    "last_change_delta_ratio",
    "change_count_20",
    "last_day_gap",
    "history_observation_count",
    "stock_code",
)
LONG_HORIZON_WINDOWS: tuple[int, ...] = (40, 60, 120, 240)
DAILY_DELTA_WINDOWS: tuple[int, ...] = (3, 5, 10, 20, 60)


class _Predictor(Protocol):
    def predict(self, features: list[dict[str, float | str]]) -> Any:
        pass


class _Classifier(Protocol):
    @property
    def classes_(self) -> Any:
        pass

    def predict_proba(self, features: list[dict[str, float | str]]) -> Any:
        pass


@dataclass(frozen=True)
class ForeignOwnershipQuantityPoint:
    stock_code: str
    base_date: date
    foreign_owned_quantity: int
    foreign_limit_quantity: int


@dataclass(frozen=True)
class ForeignOwnershipQuantityFeatureSample:
    stock_code: str
    sample_date: date
    target_quantity: int
    previous_quantity: int
    target_delta_ratio: float
    features: dict[str, float | str]


@dataclass(frozen=True)
class ForeignOwnershipQuantityPrediction:
    predicted_quantity: int
    lower_quantity: int
    upper_quantity: int
    model_version: str
    confidence_level: str
    confidence_score: float
    source: str


class ForeignOwnershipQuantityModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConstantChangeProbabilityClassifier:
    change_probability: float
    classes_: tuple[int, int] = (0, 1)

    def predict_proba(self, features: list[dict[str, float | str]]) -> list[list[float]]:
        no_change_probability = 1.0 - self.change_probability
        return [[no_change_probability, self.change_probability] for _ in features]


@dataclass(frozen=True)
class ForeignOwnershipQuantityHurdleDeltaModel:
    change_classifier: _Classifier
    change_regressor: _Predictor
    probability_threshold: float
    max_abs_delta_ratio: float = 0.25

    def predict(self, features: list[dict[str, float | str]]) -> list[float]:
        probabilities = self.change_classifier.predict_proba(features)
        positive_index = self._positive_class_index()
        raw_delta_ratios = self.change_regressor.predict(features)
        predictions: list[float] = []
        for row_index, raw_delta_ratio in enumerate(raw_delta_ratios):
            change_probability = float(probabilities[row_index][positive_index])
            if change_probability < self.probability_threshold:
                predictions.append(0.0)
                continue
            predictions.append(self._clip_delta_ratio(float(raw_delta_ratio)))
        return predictions

    def _positive_class_index(self) -> int:
        classes = list(self.change_classifier.classes_)
        return classes.index(1) if 1 in classes else len(classes) - 1

    def _clip_delta_ratio(self, value: float) -> float:
        return max(-self.max_abs_delta_ratio, min(self.max_abs_delta_ratio, value))


class ForeignOwnershipQuantityModel:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise ForeignOwnershipQuantityModelUnavailableError(
                f"Foreign ownership quantity model artifact not found: {model_path}"
            )
        payload = cast(dict[str, Any], joblib.load(model_path))
        _validate_payload(payload, model_path)
        self._model = cast(_Predictor, payload["model"])
        self._blend_alpha = float(payload["blend_alpha"])
        self._residual_abs_p90_ratio = float(payload["residual_abs_p90_ratio"])
        self._prediction_interval_abs_p90_by_stock = {
            str(stock_code): max(1, int(round(float(value))))
            for stock_code, value in dict(payload["prediction_interval_abs_p90_by_stock"]).items()
        }
        self._model_version = MODEL_VERSION
        self._minimum_history_observations = int(payload["minimum_history_observations"])
        self._release_status = str(payload["release_status"])
        self._runtime_by_stock = {
            str(stock_code): str(runtime)
            for stock_code, runtime in dict(payload.get("runtime_by_stock", {})).items()
        }
        models_by_name = dict(payload.get("models_by_name", {}))
        self._models_by_name = {
            str(model_name): cast(_Predictor, model) for model_name, model in models_by_name.items()
        }
        self._fallback_model_name = str(
            payload.get("fallback_model") or payload.get("selected_model") or ""
        )
        if self._fallback_model_name and self._fallback_model_name not in self._models_by_name:
            self._models_by_name[self._fallback_model_name] = self._model
        self._model_by_stock = {
            str(stock_code): str(model_name)
            for stock_code, model_name in dict(payload.get("model_by_stock", {})).items()
        }
        self._blend_alpha_by_stock = {
            str(stock_code): float(blend_alpha)
            for stock_code, blend_alpha in dict(payload.get("blend_alpha_by_stock", {})).items()
        }
        self._model_prediction_modes = {
            str(model_name): str(prediction_mode)
            for model_name, prediction_mode in dict(
                payload.get("model_prediction_modes", {})
            ).items()
        }

    def predict(
        self,
        stock_code: str,
        history: list[ForeignOwnershipQuantityPoint],
    ) -> ForeignOwnershipQuantityPrediction:
        if self._release_status != "promoted":
            raise ForeignOwnershipQuantityModelUnavailableError(
                "foreign ownership quantity ML model is not promoted"
            )
        points = _sorted_valid_points(stock_code, history)
        if len(points) < self._minimum_history_observations:
            raise ForeignOwnershipQuantityModelUnavailableError(
                "not enough foreign ownership quantity history"
            )
        previous_quantity = points[-1].foreign_owned_quantity
        runtime_policy = self._runtime_by_stock.get(stock_code, RUNTIME_POLICY_ML)
        if runtime_policy != RUNTIME_POLICY_ML:
            features = build_prediction_features(points)
            return _baseline_prediction_response(
                runtime_policy_quantity(previous_quantity, features, runtime_policy),
                self._model_version,
                _runtime_policy_confidence_level(runtime_policy),
                _runtime_policy_source(runtime_policy),
                self._prediction_band(stock_code, previous_quantity, points),
            )
        features = build_prediction_features(points)
        model = self._model_for_stock(stock_code)
        blend_alpha = self._blend_alpha_by_stock.get(stock_code, self._blend_alpha)
        model_name = self._model_name_for_stock(stock_code)
        raw_prediction = float(model.predict([features])[0])
        predicted_quantity = _quantity_prediction(
            previous_quantity,
            features,
            raw_prediction,
            blend_alpha,
            self._model_prediction_modes.get(
                model_name,
                PREDICTION_MODE_DELTA_RATIO,
            ),
        )
        band = self._prediction_band(stock_code, previous_quantity, points)
        confidence_score = 0.86 if self._release_status == "promoted" else 0.72
        return ForeignOwnershipQuantityPrediction(
            predicted_quantity=predicted_quantity,
            lower_quantity=max(1, predicted_quantity - band),
            upper_quantity=predicted_quantity + band,
            model_version=self._model_version,
            confidence_level="AI_FOREIGN_OWNED_QUANTITY_ML",
            confidence_score=confidence_score,
            source="KMARKET_AI_FOREIGN_OWNED_QUANTITY_ML",
        )

    def _model_for_stock(self, stock_code: str) -> _Predictor:
        model_name = self._model_name_for_stock(stock_code)
        if model_name in self._models_by_name:
            return self._models_by_name[model_name]
        return self._model

    def _model_name_for_stock(self, stock_code: str) -> str:
        return self._model_by_stock.get(stock_code, self._fallback_model_name)

    def _prediction_band(
        self,
        stock_code: str,
        previous_quantity: int,
        points: list[ForeignOwnershipQuantityPoint],
    ) -> int:
        calibrated_band = self._prediction_interval_abs_p90_by_stock.get(stock_code)
        if calibrated_band is not None:
            # 과거 검증오차를 상한으로 두고 최신 변동성 국면의 실제 90분위수를 반영한다.
            return min(calibrated_band, _recent_abs_delta_p90_quantity(points))
        return max(1, _clip_quantity(previous_quantity * self._residual_abs_p90_ratio))


def _baseline_prediction_response(
    previous_quantity: int,
    model_version: str,
    confidence_level: str,
    source: str,
    prediction_band: int,
) -> ForeignOwnershipQuantityPrediction:
    band = max(1, prediction_band)
    return ForeignOwnershipQuantityPrediction(
        predicted_quantity=previous_quantity,
        lower_quantity=max(0, previous_quantity - band),
        upper_quantity=previous_quantity + band,
        model_version=model_version,
        confidence_level=confidence_level,
        confidence_score=0.74,
        source=source,
    )


def _quantity_prediction(
    previous_quantity: int,
    features: dict[str, float | str],
    raw_prediction: float,
    blend_alpha: float,
    prediction_mode: str,
) -> int:
    if prediction_mode.startswith(PREDICTION_MODE_RESIDUAL_PREFIX):
        base_policy = prediction_mode.removeprefix(PREDICTION_MODE_RESIDUAL_PREFIX)
        return _clip_quantity(
            runtime_policy_quantity(previous_quantity, features, base_policy)
            + blend_alpha * raw_prediction
        )
    if prediction_mode == PREDICTION_MODE_DELTA_QUANTITY:
        return _clip_quantity(previous_quantity + blend_alpha * raw_prediction)
    if prediction_mode == PREDICTION_MODE_TARGET_QUANTITY:
        return _clip_quantity(
            previous_quantity + blend_alpha * (raw_prediction - previous_quantity)
        )
    if prediction_mode == PREDICTION_MODE_LOG_DELTA_RATIO:
        return _clip_quantity(
            previous_quantity * exp(_clip_log_delta(blend_alpha * raw_prediction))
        )
    return _clip_quantity(previous_quantity * (1.0 + blend_alpha * raw_prediction))


def build_training_samples(
    points: list[ForeignOwnershipQuantityPoint],
) -> list[ForeignOwnershipQuantityFeatureSample]:
    samples: list[ForeignOwnershipQuantityFeatureSample] = []
    by_stock: dict[str, list[ForeignOwnershipQuantityPoint]] = {}
    for point in points:
        if point.foreign_owned_quantity > 0 and point.stock_code.isdigit():
            by_stock.setdefault(point.stock_code, []).append(point)

    for stock_code, stock_points in by_stock.items():
        ordered_points = sorted(stock_points, key=lambda point: point.base_date)
        change_indices: list[int] = []
        change_delta_ratios: list[float] = []
        change_interval_day_sum = 0.0
        change_interval_count = 0
        previous_change_date: date | None = None
        for history_end_index in range(len(ordered_points) - 1):
            if history_end_index > 0:
                previous = ordered_points[history_end_index - 1]
                current = ordered_points[history_end_index]
                if current.foreign_owned_quantity != previous.foreign_owned_quantity:
                    change_indices.append(history_end_index)
                    change_delta_ratios.append(
                        _delta_ratio(
                            current.foreign_owned_quantity,
                            previous.foreign_owned_quantity,
                        )
                    )
                    if previous_change_date is not None:
                        change_interval_day_sum += (current.base_date - previous_change_date).days
                        change_interval_count += 1
                    previous_change_date = current.base_date

            history_observation_count = history_end_index + 1
            if history_observation_count < MIN_HISTORY_OBSERVATIONS:
                continue

            target = ordered_points[history_end_index + 1]
            previous_quantity = ordered_points[history_end_index].foreign_owned_quantity
            target_delta_ratio = _delta_ratio(target.foreign_owned_quantity, previous_quantity)
            samples.append(
                ForeignOwnershipQuantityFeatureSample(
                    stock_code=stock_code,
                    sample_date=target.base_date,
                    target_quantity=target.foreign_owned_quantity,
                    previous_quantity=previous_quantity,
                    target_delta_ratio=target_delta_ratio,
                    features=_build_training_features(
                        ordered_points,
                        history_end_index,
                        change_indices,
                        change_delta_ratios,
                        change_interval_day_sum,
                        change_interval_count,
                    ),
                )
            )
    return sorted(samples, key=lambda sample: (sample.sample_date, sample.stock_code))


def _build_training_features(
    points: list[ForeignOwnershipQuantityPoint],
    latest_index: int,
    change_indices: list[int],
    change_delta_ratios: list[float],
    change_interval_day_sum: float,
    change_interval_count: int,
) -> dict[str, float | str]:
    # 대량 학습에서는 prefix slice/sort 없이 rolling 상태로 동일 feature를 만든다.
    latest_point = points[latest_index]
    latest = float(latest_point.foreign_owned_quantity)
    latest_limit_quantity = float(max(1, latest_point.foreign_limit_quantity))
    features: dict[str, float | str] = {
        "stock_code": latest_point.stock_code,
        "foreign_limit_quantity": latest_limit_quantity,
        "foreign_limit_exhaustion_rate": latest * 100.0 / latest_limit_quantity,
        "foreign_limit_remaining_quantity": latest_limit_quantity - latest,
        "foreign_limit_remaining_rate": (latest_limit_quantity - latest)
        * 100.0
        / latest_limit_quantity,
        "history_observation_count": float(latest_index + 1),
        "last_day_gap": float(
            max(1, (latest_point.base_date - points[latest_index - 1].base_date).days)
        ),
        "latest_month": float(latest_point.base_date.month),
        "latest_day_of_month": float(latest_point.base_date.day),
        "latest_day_of_week": float(latest_point.base_date.weekday()),
        "latest_ordinal_mod_7": float(latest_point.base_date.toordinal() % 7),
        "latest_ordinal_mod_30": float(latest_point.base_date.toordinal() % 30),
    }
    for lag in (1, 2, 3, 4, 5, 10, 20):
        lag_value = float(points[latest_index - lag + 1].foreign_owned_quantity)
        features[f"lag_{lag}"] = lag_value
        features[f"delta_{lag}"] = latest - lag_value
        features[f"pct_delta_{lag}"] = _delta_ratio(latest, lag_value)
    for window in (3, 5, 10, 20):
        window_values = [
            float(point.foreign_owned_quantity)
            for point in points[latest_index - window + 1 : latest_index + 1]
        ]
        features[f"mean_{window}"] = mean(window_values)
        if window in (5, 20):
            features[f"std_{window}"] = pstdev(window_values)
    last_20 = [
        float(point.foreign_owned_quantity)
        for point in points[latest_index - 19 : latest_index + 1]
    ]
    features["min_20"] = min(last_20)
    features["max_20"] = max(last_20)
    features["range_20"] = max(last_20) - min(last_20)
    feature_window_start = max(0, latest_index - max(LONG_HORIZON_WINDOWS) + 1)
    recent_values = [
        float(point.foreign_owned_quantity)
        for point in points[feature_window_start : latest_index + 1]
    ]
    features.update(_long_horizon_features(recent_values))
    features.update(_daily_delta_distribution_features(recent_values))
    features.update(
        _rolling_change_interval_features(
            points,
            latest_index,
            change_indices,
            change_delta_ratios,
            change_interval_day_sum,
            change_interval_count,
        )
    )
    return features


def _rolling_change_interval_features(
    points: list[ForeignOwnershipQuantityPoint],
    latest_index: int,
    change_indices: list[int],
    change_delta_ratios: list[float],
    change_interval_day_sum: float,
    change_interval_count: int,
) -> dict[str, float]:
    if not change_indices:
        return {
            "observations_since_last_change": float(latest_index + 1),
            "days_since_last_change": float(
                (points[latest_index].base_date - points[0].base_date).days
            ),
            "prior_change_count": 0.0,
            "prior_change_rate": 0.0,
            "mean_days_between_changes": 0.0,
            "last_change_delta_ratio": 0.0,
            "change_count_20": 0.0,
        }

    last_change_index = change_indices[-1]
    first_recent_change_index = bisect_right(change_indices, latest_index - 20)
    return {
        "observations_since_last_change": float(latest_index - last_change_index),
        "days_since_last_change": float(
            (points[latest_index].base_date - points[last_change_index].base_date).days
        ),
        "prior_change_count": float(len(change_indices)),
        "prior_change_rate": len(change_indices) / max(1.0, latest_index),
        "mean_days_between_changes": (
            change_interval_day_sum / change_interval_count if change_interval_count else 0.0
        ),
        "last_change_delta_ratio": change_delta_ratios[-1],
        "change_count_20": float(len(change_indices) - first_recent_change_index),
    }


def build_prediction_features(
    history: list[ForeignOwnershipQuantityPoint],
) -> dict[str, float | str]:
    points = sorted(history, key=lambda point: point.base_date)
    if len(points) < MIN_HISTORY_OBSERVATIONS:
        raise ValueError(
            "foreign ownership quantity feature generation requires at least 20 observations"
        )
    values = [float(point.foreign_owned_quantity) for point in points]
    latest = values[-1]
    latest_limit_quantity = float(max(1, points[-1].foreign_limit_quantity))
    features: dict[str, float | str] = {
        "stock_code": points[-1].stock_code,
        "foreign_limit_quantity": latest_limit_quantity,
        "foreign_limit_exhaustion_rate": latest * 100.0 / latest_limit_quantity,
        "foreign_limit_remaining_quantity": latest_limit_quantity - latest,
        "foreign_limit_remaining_rate": (latest_limit_quantity - latest)
        * 100.0
        / latest_limit_quantity,
        "history_observation_count": float(len(points)),
        "last_day_gap": float(max(1, (points[-1].base_date - points[-2].base_date).days)),
        "latest_month": float(points[-1].base_date.month),
        "latest_day_of_month": float(points[-1].base_date.day),
        "latest_day_of_week": float(points[-1].base_date.weekday()),
        "latest_ordinal_mod_7": float(points[-1].base_date.toordinal() % 7),
        "latest_ordinal_mod_30": float(points[-1].base_date.toordinal() % 30),
    }
    for lag in (1, 2, 3, 4, 5, 10, 20):
        lag_value = values[-lag]
        features[f"lag_{lag}"] = lag_value
        features[f"delta_{lag}"] = latest - lag_value
        features[f"pct_delta_{lag}"] = _delta_ratio(latest, lag_value)
    for window in (3, 5, 10, 20):
        window_values = values[-window:]
        features[f"mean_{window}"] = mean(window_values)
        if window in (5, 20):
            features[f"std_{window}"] = pstdev(window_values)
    last_20 = values[-20:]
    features["min_20"] = min(last_20)
    features["max_20"] = max(last_20)
    features["range_20"] = max(last_20) - min(last_20)
    features.update(_long_horizon_features(values))
    features.update(_daily_delta_distribution_features(values))
    features.update(_change_interval_features(points))
    return features


def _long_horizon_features(values: list[float]) -> dict[str, float]:
    latest = values[-1]
    features: dict[str, float] = {}
    for window in LONG_HORIZON_WINDOWS:
        lag_index = max(0, len(values) - window)
        lag_value = values[lag_index]
        distance = max(1, len(values) - lag_index - 1)
        window_values = values[-window:] if len(values) >= window else values
        features[f"lag_{window}"] = lag_value
        features[f"delta_{window}"] = latest - lag_value
        features[f"pct_delta_{window}"] = _delta_ratio(latest, lag_value)
        features[f"mean_{window}"] = mean(window_values)
        features[f"mean_delta_{window}"] = (latest - lag_value) / distance
    return features


def _daily_delta_distribution_features(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        return _empty_daily_delta_distribution_features()

    deltas = [current - previous for previous, current in zip(values, values[1:], strict=False)]
    features: dict[str, float] = {}
    for lag in range(1, 6):
        features[f"daily_delta_lag_{lag}"] = deltas[-lag] if len(deltas) >= lag else 0.0
    for window in DAILY_DELTA_WINDOWS:
        recent = deltas[-window:] if len(deltas) >= window else deltas
        abs_recent = [abs(value) for value in recent]
        positive_count = sum(1 for value in recent if value > 0)
        negative_count = sum(1 for value in recent if value < 0)
        zero_count = len(recent) - positive_count - negative_count
        features[f"daily_delta_mean_{window}"] = mean(recent)
        features[f"daily_delta_median_{window}"] = median(recent)
        features[f"daily_delta_std_{window}"] = pstdev(recent) if len(recent) > 1 else 0.0
        features[f"daily_delta_abs_mean_{window}"] = mean(abs_recent)
        features[f"daily_delta_positive_share_{window}"] = positive_count / len(recent)
        features[f"daily_delta_negative_share_{window}"] = negative_count / len(recent)
        features[f"daily_delta_zero_share_{window}"] = zero_count / len(recent)
    features["daily_delta_acceleration_3_20"] = (
        features["daily_delta_mean_3"] - features["daily_delta_mean_20"]
    )
    features["daily_delta_acceleration_5_60"] = (
        features["daily_delta_mean_5"] - features["daily_delta_mean_60"]
    )
    return features


def _empty_daily_delta_distribution_features() -> dict[str, float]:
    features = {f"daily_delta_lag_{lag}": 0.0 for lag in range(1, 6)}
    for window in DAILY_DELTA_WINDOWS:
        features[f"daily_delta_mean_{window}"] = 0.0
        features[f"daily_delta_median_{window}"] = 0.0
        features[f"daily_delta_std_{window}"] = 0.0
        features[f"daily_delta_abs_mean_{window}"] = 0.0
        features[f"daily_delta_positive_share_{window}"] = 0.0
        features[f"daily_delta_negative_share_{window}"] = 0.0
        features[f"daily_delta_zero_share_{window}"] = 1.0
    features["daily_delta_acceleration_3_20"] = 0.0
    features["daily_delta_acceleration_5_60"] = 0.0
    return features


def _change_interval_features(points: list[ForeignOwnershipQuantityPoint]) -> dict[str, float]:
    change_indices: list[int] = []
    change_delta_ratios: list[float] = []
    for index in range(1, len(points)):
        previous = points[index - 1]
        current = points[index]
        if current.foreign_owned_quantity != previous.foreign_owned_quantity:
            change_indices.append(index)
            change_delta_ratios.append(
                _delta_ratio(current.foreign_owned_quantity, previous.foreign_owned_quantity)
            )

    latest_index = len(points) - 1
    if not change_indices:
        return {
            "observations_since_last_change": float(len(points)),
            "days_since_last_change": float((points[-1].base_date - points[0].base_date).days),
            "prior_change_count": 0.0,
            "prior_change_rate": 0.0,
            "mean_days_between_changes": 0.0,
            "last_change_delta_ratio": 0.0,
            "change_count_20": 0.0,
        }

    last_change_index = change_indices[-1]
    change_dates = [points[index].base_date for index in change_indices]
    intervals = [
        (current - previous).days
        for previous, current in zip(change_dates, change_dates[1:], strict=False)
    ]
    return {
        "observations_since_last_change": float(latest_index - last_change_index),
        "days_since_last_change": float(
            (points[-1].base_date - points[last_change_index].base_date).days
        ),
        "prior_change_count": float(len(change_indices)),
        "prior_change_rate": len(change_indices) / max(1.0, len(points) - 1),
        "mean_days_between_changes": mean(intervals) if intervals else 0.0,
        "last_change_delta_ratio": change_delta_ratios[-1],
        "change_count_20": float(sum(1 for index in change_indices if latest_index - index < 20)),
    }


def baseline_prediction(sample: ForeignOwnershipQuantityFeatureSample) -> int:
    return sample.previous_quantity


def runtime_policy_quantity(
    previous_quantity: int,
    features: dict[str, float | str],
    policy: str,
) -> int:
    if policy in {RUNTIME_POLICY_BASELINE, RUNTIME_POLICY_ML}:
        return previous_quantity
    if policy == RUNTIME_POLICY_STALE_GUARDED_MEAN_DELTA_20:
        if _feature_float(features, "observations_since_last_change") > 5.0:
            return previous_quantity
        return _clip_quantity(previous_quantity + 0.25 * _mean_delta(features, 20))
    if policy == RUNTIME_POLICY_MEAN_DELTA_20:
        return _clip_quantity(previous_quantity + _mean_delta(features, 20))
    if policy == RUNTIME_POLICY_MEDIAN_MULTI_DELTA:
        deltas = [
            _feature_float(features, "lag_1") - _feature_float(features, "lag_2"),
            (_feature_float(features, "lag_1") - _feature_float(features, "lag_3")) / 2.0,
            (_feature_float(features, "lag_1") - _feature_float(features, "lag_5")) / 4.0,
            (_feature_float(features, "lag_1") - _feature_float(features, "lag_10")) / 9.0,
            _mean_delta(features, 20),
        ]
        return _clip_quantity(previous_quantity + median(deltas))
    if policy == RUNTIME_POLICY_MICRO_MEDIAN_DELTA_3:
        deltas = [
            _feature_float(features, "lag_1") - _feature_float(features, "lag_2"),
            _feature_float(features, "lag_2") - _feature_float(features, "lag_3"),
            _feature_float(features, "lag_3") - _feature_float(features, "lag_4"),
        ]
        return _clip_quantity(previous_quantity + 0.1 * median(deltas))
    return previous_quantity


def blended_model_prediction(
    sample: ForeignOwnershipQuantityFeatureSample,
    predicted_delta_ratio: float,
    blend_alpha: float,
) -> int:
    return _clip_quantity(sample.previous_quantity * (1.0 + blend_alpha * predicted_delta_ratio))


def regression_metrics(actual: list[int], predicted: list[int]) -> dict[str, float]:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths must match")
    if not actual:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    errors = [abs(a - p) for a, p in zip(actual, predicted, strict=True)]
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted, strict=True)]
    percentage_errors = [error / max(1, abs(a)) for error, a in zip(errors, actual, strict=True)]
    return {
        "mae": mean(errors),
        "rmse": sqrt(mean(squared_errors)),
        "mape": mean(percentage_errors),
    }


def residual_abs_p90_ratio(actual: list[int], predicted: list[int]) -> float:
    if not actual:
        return 0.001
    ratios = sorted(abs(a - p) / max(1, a) for a, p in zip(actual, predicted, strict=True))
    index = min(len(ratios) - 1, int(len(ratios) * 0.9))
    return max(0.0001, ratios[index])


def _recent_abs_delta_p90_quantity(
    points: list[ForeignOwnershipQuantityPoint],
) -> int:
    recent_points = points[-60:]
    absolute_deltas = sorted(
        abs(current.foreign_owned_quantity - previous.foreign_owned_quantity)
        for previous, current in zip(recent_points, recent_points[1:], strict=False)
    )
    if not absolute_deltas:
        return 1
    index = min(len(absolute_deltas) - 1, int(len(absolute_deltas) * 0.9))
    return max(1, absolute_deltas[index])


def _sorted_valid_points(
    stock_code: str,
    history: list[ForeignOwnershipQuantityPoint],
) -> list[ForeignOwnershipQuantityPoint]:
    return sorted(
        (
            point
            for point in history
            if point.stock_code == stock_code and point.foreign_owned_quantity > 0
        ),
        key=lambda point: point.base_date,
    )


def _validate_payload(payload: dict[str, Any], model_path: Path) -> None:
    required_keys = {
        "schema_version",
        "model_version",
        "model",
        "blend_alpha",
        "residual_abs_p90_ratio",
        "prediction_interval_abs_p90_by_stock",
        "minimum_history_observations",
        "release_status",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        joined_keys = ", ".join(missing_keys)
        raise ForeignOwnershipQuantityModelUnavailableError(
            f"Foreign ownership quantity model artifact is missing required keys: "
            f"{joined_keys} ({model_path})"
        )
    if payload["schema_version"] != MODEL_SCHEMA_VERSION:
        raise ForeignOwnershipQuantityModelUnavailableError(
            f"Unsupported foreign ownership quantity model schema: {payload['schema_version']}"
        )


def _delta_ratio(current: float, previous: float) -> float:
    return (current - previous) / max(1.0, abs(previous))


def _mean_delta(features: dict[str, float | str], window: int) -> float:
    return (_feature_float(features, "lag_1") - _feature_float(features, f"lag_{window}")) / max(
        1, window - 1
    )


def _feature_float(features: dict[str, float | str], name: str) -> float:
    return float(features.get(name, 0.0))


def _runtime_policy_confidence_level(policy: str) -> str:
    if policy == RUNTIME_POLICY_BASELINE:
        return "AI_FOREIGN_OWNED_QUANTITY_STOCK_GUARDED_BASELINE"
    return "AI_FOREIGN_OWNED_QUANTITY_STOCK_GUARDED_HEURISTIC"


def _runtime_policy_source(policy: str) -> str:
    if policy == RUNTIME_POLICY_BASELINE:
        return "KMARKET_AI_FOREIGN_OWNED_QUANTITY_STOCK_GUARDED_BASELINE"
    return f"KMARKET_AI_FOREIGN_OWNED_QUANTITY_{policy.upper()}"


def _clip_quantity(value: float) -> int:
    return max(1, round(value))


def _clip_log_delta(value: float) -> float:
    return max(-0.5, min(0.5, value))
