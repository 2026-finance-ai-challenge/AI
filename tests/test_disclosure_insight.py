import asyncio
from types import SimpleNamespace

from k_market_ai.core.config import Settings
from k_market_ai.rag.application.disclosure_insight import (
    DisclosureInsightService,
    FilingEvidence,
)


def test_disclosure_insight_uses_only_structured_evidence_without_storage() -> None:
    responses = FakeResponses()
    service = DisclosureInsightService(
        SimpleNamespace(responses=responses),
        Settings(
            environment="test",
            news_model="test-summary-model",
            filing_summary_prompt_version="filing-summary-test-v2",
        ),
    )

    result = asyncio.run(
        service.summarize(
            "20260823800001",
            "Major Business Report",
            (FilingEvidence("S1", "Investment", "The company approved a new facility."),),
        )
    )

    assert result.what == "The company approved a new facility."
    assert result.evidence_ids == ("S1",)
    assert result.model == "test-summary-model"
    assert result.prompt_version == "filing-summary-test-v2"
    assert responses.arguments["store"] is False
    assert "The company approved a new facility." in str(responses.arguments["input"])


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        return SimpleNamespace(
            output_parsed=SimpleNamespace(
                what="The company approved a new facility.",
                why="The filing states that the facility will expand capacity.",
                impact="The facility may increase production capacity.",
                evidence_ids=("S1",),
                sufficient_evidence=True,
                refusal_reason=None,
            )
        )
