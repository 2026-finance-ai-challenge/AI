from __future__ import annotations

import hashlib
import importlib
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from k_market_ai.news.domain import MarketImpact, NewsImportance, NewsSentiment

EXPECTED_HANA_COMMIT = "ab82ccc51cb096872f9a110a85c027a4158a147f"
EXPECTED_FILE_SHA256 = {
    "src/hannah_montana_ai/model_store/financial_nlp_ml.joblib": (
        "04bb18037d28c59c487779531c90db5faa2e2136a3ca1dfe1d7af1a781ad6157"
    ),
    "src/hannah_montana_ai/model_store/k_fnspid_impact_news_ml.joblib": (
        "df852dcddb8e76436f415153fe34e86b9671bfc2134d78be648df513acb6f3f6"
    ),
    "src/hannah_montana_ai/model_store/k_fnspid_impact_disclosure_ml.joblib": (
        "a1b5a021ba47cff72300e77cf694cf3aa093b232efeecd9be14627ccb2e04822"
    ),
    "reports/k-fnspid-impact-news-training-report.json": (
        "c923702da9d221cd443dddc62df43c767c4cbbe851f249cc19b32f2fe5d016f6"
    ),
    "reports/k-fnspid-impact-disclosure-training-report.json": (
        "22a5eb0c47188d2b83e444b20dfa7854a79de883d8cb2726340d54409fa67a41"
    ),
    "reports/kf-deberta-sentiment-training-report.json": (
        "78c6db262e9263c84b32bd580c30b81335baea56ea210057fbb36edb58039a01"
    ),
    "reports/korean-finance-sentiment-benchmark.json": (
        "996b6d0bcbd03a508dd36d7ceb2ab4135de1deaffa15854a987137147c5b71f9"
    ),
    "src/hannah_montana_ai/model_store/kf_deberta_sentiment/adapter_model.safetensors": (
        "506a4290af390f9ebd3a3cabc8ae592e6c4c53837d44f1fb821c86819dd81c88"
    ),
}


class NewsClassifierUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class NewsSignals:
    event_type: str
    sentiment: NewsSentiment
    importance: NewsImportance
    market_impact: MarketImpact
    event_confidence: float
    sentiment_confidence: float
    importance_confidence: float
    market_impact_confidence: float
    market_impact_level: NewsImportance
    market_impact_score: float
    model_version: str


class NewsSignalClassifier(Protocol):
    def classify(
        self,
        title: str,
        paragraphs: tuple[str, ...],
        candidate_companies: tuple[str, ...],
        source_type: Literal["NEWS", "DISCLOSURE"] = "NEWS",
    ) -> NewsSignals: ...


class HanaNewsSignalClassifier:
    def __init__(
        self,
        project_root: Path,
        *,
        expected_commit: str = EXPECTED_HANA_COMMIT,
        runtime_environment: str = "local",
    ) -> None:
        self._root = project_root.resolve(strict=True)
        self._expected_commit = expected_commit
        self._runtime_environment = runtime_environment
        self._lock = threading.Lock()
        self._models: tuple[Any, Any, Any, Any] | None = None

    def classify(
        self,
        title: str,
        paragraphs: tuple[str, ...],
        candidate_companies: tuple[str, ...],
        source_type: Literal["NEWS", "DISCLOSURE"] = "NEWS",
    ) -> NewsSignals:
        model, sentiment_model, news_impact_model, disclosure_impact_model = self._load()
        impact_model = disclosure_impact_model if source_type == "DISCLOSURE" else news_impact_model
        text = "\n".join((title, *paragraphs)).strip()
        target = candidate_companies[0] if candidate_companies else ""

        event_probabilities = cast(
            dict[str, float], model.event_tag_probabilities(text, source_type)
        )
        event_type = max(event_probabilities, key=event_probabilities.__getitem__)
        sentiment_probabilities = None
        if sentiment_model.enabled:
            sentiment_probabilities = sentiment_model.probabilities(text, source_type, target)
        if sentiment_probabilities is None:
            sentiment_probabilities = cast(dict[str, float], model.sentiment_probabilities(text))
        impact_prediction = impact_model.predict(text, source_type)
        if impact_prediction is None:
            raise NewsClassifierUnavailable("The verified Hana classifier returned no result.")

        sentiment_label = max(sentiment_probabilities, key=sentiment_probabilities.__getitem__)
        importance_probabilities = cast(
            dict[str, float], model.importance_probabilities(text, source_type)
        )
        importance_label = max(importance_probabilities, key=importance_probabilities.__getitem__)
        direction = {
            "POSITIVE": MarketImpact.POSITIVE,
            "NEGATIVE": MarketImpact.NEGATIVE,
            "NEUTRAL": MarketImpact.NEUTRAL,
        }[sentiment_label]
        versions = (
            str(model.version),
            str(impact_prediction.model_version),
        )
        bundle_digest = hashlib.sha256("|".join(versions).encode()).hexdigest()[:12]
        return NewsSignals(
            event_type=event_type,
            sentiment=NewsSentiment(sentiment_label),
            importance=NewsImportance(importance_label),
            market_impact=direction,
            event_confidence=float(event_probabilities[event_type]),
            sentiment_confidence=float(sentiment_probabilities[sentiment_label]),
            importance_confidence=float(importance_probabilities[importance_label]),
            market_impact_confidence=min(
                float(sentiment_probabilities[sentiment_label]),
                float(impact_prediction.confidence),
            ),
            market_impact_level=NewsImportance(str(impact_prediction.importance)),
            market_impact_score=float(impact_prediction.materiality_score),
            model_version=f"hana-finance-{bundle_digest}",
        )

    def _load(self) -> tuple[Any, Any, Any, Any]:
        if self._models is not None:
            return self._models
        with self._lock:
            if self._models is not None:
                return self._models
            self._verify_source_and_artifacts()
            source_root = self._root / "src"
            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
            model_module = importlib.import_module("hannah_montana_ai.services.model")
            sentiment_module = importlib.import_module(
                "hannah_montana_ai.services.transformer_sentiment_model"
            )
            impact_module = importlib.import_module(
                "hannah_montana_ai.services.market_impact_model"
            )
            model = model_module.MachineLearningFinancialNlpModel(
                self._root / "src/hannah_montana_ai/model_store/financial_nlp_ml.joblib"
            )
            sentiment = sentiment_module.KfDebertaSentimentModel(
                self._root / "src/hannah_montana_ai/model_store/kf_deberta_sentiment",
                self._root / "reports/kf-deberta-sentiment-training-report.json",
                self._root / "reports/korean-finance-sentiment-benchmark.json",
                self._root
                / "artifacts/pretraining/kf-deberta-k-fnspid-v4-dapt-temporal-v2/merged_fp32",
                release_current_path=None,
                project_root=self._root,
                runtime_environment=self._runtime_environment,
                release_required=False,
            )
            news_impact = impact_module.KFnspidMarketImpactModel(
                self._root / "src/hannah_montana_ai/model_store/k_fnspid_impact_news_ml.joblib",
                self._root / "reports/k-fnspid-impact-news-training-report.json",
                "NEWS",
            )
            disclosure_impact = impact_module.KFnspidMarketImpactModel(
                self._root
                / "src/hannah_montana_ai/model_store/k_fnspid_impact_disclosure_ml.joblib",
                self._root / "reports/k-fnspid-impact-disclosure-training-report.json",
                "DISCLOSURE",
            )
            if not news_impact.enabled or not disclosure_impact.enabled:
                raise NewsClassifierUnavailable(
                    "The verified Hana K-FNSPID artifact did not pass its deployment gate."
                )
            self._models = (model, sentiment, news_impact, disclosure_impact)
            return self._models

    def _verify_source_and_artifacts(self) -> None:
        if _git_commit(self._root) != self._expected_commit:
            raise NewsClassifierUnavailable("The mounted Hana source revision is not approved.")
        for relative, expected in EXPECTED_FILE_SHA256.items():
            path = self._root / relative
            if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
                raise NewsClassifierUnavailable(
                    f"The mounted Hana artifact failed integrity verification: {relative}"
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    git_dir = root / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    raise NewsClassifierUnavailable("The mounted Hana Git revision cannot be verified.")
