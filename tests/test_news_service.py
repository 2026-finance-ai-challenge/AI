import asyncio
from types import SimpleNamespace

from k_market_ai.core.config import Settings
from k_market_ai.news.classifier import NewsSignals
from k_market_ai.news.domain import (
    MarketImpact,
    NewsImportance,
    NewsSentiment,
    TermEvidence,
)
from k_market_ai.news.service import NewsIntelligenceService


def test_news_analysis_uses_structured_response_without_storage() -> None:
    responses = FakeResponses("analysis")
    service = NewsIntelligenceService(
        SimpleNamespace(responses=responses),
        Settings(environment="test", news_model="test-news-model"),
        FakeClassifier(),
    )

    result = asyncio.run(
        service.analyze(
            "삼성전자 신제품 공개",
            ("삼성전자가 신제품을 공개했다.", "회사는 수요 증가를 기대했다."),
            ("Samsung Electronics",),
        )
    )

    assert result.english_title == "Samsung Electronics unveils new product"
    assert result.translated_paragraphs == (
        "Samsung Electronics unveiled a new product.",
        "The company expects demand to increase.",
    )
    assert result.model == "test-news-model+hana-test-v1"
    assert result.event_type == "PRODUCT_LAUNCH"
    assert result.importance == NewsImportance.HIGH
    assert result.market_impact_importance == NewsImportance.MEDIUM
    assert result.market_impact_score == 0.55
    assert responses.arguments["store"] is False
    assert "삼성전자가 신제품을 공개했다." in str(responses.arguments["input"])


def test_term_explanation_passes_hashed_safety_identifier_and_evidence() -> None:
    responses = FakeResponses("term")
    service = NewsIntelligenceService(
        SimpleNamespace(responses=responses),
        Settings(environment="test", term_prompt_version="term-test-v2"),
        FakeClassifier(),
    )
    safety_identifier = "a" * 64

    result = asyncio.run(
        service.explain_term(
            "유상증자",
            "회사는 신주를 발행하는 유상증자를 결정했다.",
            (
                TermEvidence(
                    id="G1",
                    title="Rights offering",
                    content="A company issues new shares in exchange for payment.",
                    source_url="https://example.test/glossary/rights-offering",
                ),
            ),
            safety_identifier,
        )
    )

    assert result.sufficient_evidence is True
    assert result.evidence_ids == ("G1",)
    assert result.prompt_version == "term-test-v2"
    assert responses.arguments["store"] is False
    assert responses.arguments["safety_identifier"] == safety_identifier


class FakeResponses:
    def __init__(self, response_type: str) -> None:
        self.response_type = response_type
        self.arguments: dict[str, object] = {}

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        if self.response_type == "analysis":
            parsed = SimpleNamespace(
                english_title="Samsung Electronics unveils new product",
                translated_paragraphs=(
                    "Samsung Electronics unveiled a new product.",
                    "The company expects demand to increase.",
                ),
                what="The company unveiled a new product.",
                why="The source does not state a reason for the launch.",
                impact="The company expects demand to increase.",
                event_type="PRODUCT_LAUNCH",
                sentiment="POSITIVE",
                importance="MEDIUM",
                market_impact="POSITIVE",
                event_confidence=0.97,
                sentiment_confidence=0.82,
                importance_confidence=0.76,
                market_impact_confidence=0.72,
            )
        else:
            parsed = SimpleNamespace(
                normalized_term="rights offering",
                definition="A company issues new shares in exchange for payment.",
                contextual_meaning="The company decided to raise capital by issuing new shares.",
                evidence_ids=("G1",),
                confidence=0.94,
                review_required=False,
                sufficient_evidence=True,
                refusal_reason=None,
            )
        return SimpleNamespace(output_parsed=parsed)


class FakeClassifier:
    def classify(
        self,
        title: str,
        paragraphs: tuple[str, ...],
        candidate_companies: tuple[str, ...],
    ) -> NewsSignals:
        assert title
        assert paragraphs
        del candidate_companies
        return NewsSignals(
            event_type="PRODUCT_LAUNCH",
            sentiment=NewsSentiment.POSITIVE,
            importance=NewsImportance.HIGH,
            market_impact=MarketImpact.POSITIVE,
            event_confidence=0.91,
            sentiment_confidence=0.89,
            importance_confidence=0.82,
            market_impact_confidence=0.78,
            market_impact_level=NewsImportance.MEDIUM,
            market_impact_score=0.55,
            model_version="hana-test-v1",
        )
