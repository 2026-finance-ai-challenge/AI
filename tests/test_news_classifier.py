from types import SimpleNamespace

import pytest

from k_market_ai.news.classifier import (
    HanaNewsSignalClassifier,
    NewsClassifierUnavailable,
    _git_commit,
)
from k_market_ai.news.domain import MarketImpact, NewsImportance, NewsSentiment


class FakeFinancialModel:
    version = "financial-v1"

    def event_tag_probabilities(self, text: str, source_type: str) -> dict[str, float]:
        assert text.startswith("제목")
        assert source_type == "NEWS"
        return {"EARNINGS": 0.83, "GENERAL_MARKET": 0.17}

    def sentiment_probabilities(self, text: str) -> dict[str, float]:
        assert text
        return {"NEGATIVE": 0.05, "NEUTRAL": 0.15, "POSITIVE": 0.8}

    def importance_probabilities(self, text: str, source_type: str) -> dict[str, float]:
        assert text and source_type == "NEWS"
        return {"LOW": 0.05, "MEDIUM": 0.15, "HIGH": 0.7, "CRITICAL": 0.1}


class FakeImpactModel:
    def predict(self, text: str, source_type: str) -> SimpleNamespace:
        assert text and source_type == "NEWS"
        return SimpleNamespace(
            importance="MEDIUM",
            confidence=0.62,
            materiality_score=0.51,
            model_version="impact-v1",
        )


class DisabledTransformer:
    enabled = False


def test_hana_classifier_keeps_semantic_and_market_impact_signals_separate(tmp_path) -> None:
    classifier = HanaNewsSignalClassifier(tmp_path, expected_commit="a" * 40)
    classifier._models = (
        FakeFinancialModel(),
        DisabledTransformer(),
        FakeImpactModel(),
        FakeImpactModel(),
    )

    result = classifier.classify("제목", ("본문",), ("Company",))

    assert result.event_type == "EARNINGS"
    assert result.sentiment == NewsSentiment.POSITIVE
    assert result.importance == NewsImportance.HIGH
    assert result.market_impact == MarketImpact.POSITIVE
    assert result.market_impact_level == NewsImportance.MEDIUM
    assert result.market_impact_confidence == 0.62
    assert result.model_version.startswith("hana-finance-")


def test_hana_source_revision_and_missing_artifact_fail_closed(tmp_path) -> None:
    git = tmp_path / ".git"
    reference = git / "refs/heads/main"
    reference.parent.mkdir(parents=True)
    reference.write_text("b" * 40 + "\n", encoding="utf-8")
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    assert _git_commit(tmp_path) == "b" * 40

    classifier = HanaNewsSignalClassifier(tmp_path, expected_commit="b" * 40)
    with pytest.raises(NewsClassifierUnavailable, match="integrity verification"):
        classifier.classify("제목", ("본문",), ())
