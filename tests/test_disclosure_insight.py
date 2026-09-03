import asyncio
from types import SimpleNamespace

import pytest

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
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
    assert responses.arguments["reasoning"] == {"effort": "low"}
    assert result.what_ko == "회사가 새 시설을 승인했다."
    assert responses.arguments["text"] == {"verbosity": "low"}
    assert "The company approved a new facility." in str(responses.arguments["input"])


def test_disclosure_insight_rejects_hangul_in_english_summary() -> None:
    service = DisclosureInsightService(
        SimpleNamespace(responses=HangulResponses()),
        Settings(environment="test", news_model="test-summary-model"),
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            service.summarize(
                "20260823800001",
                "Major Business Report",
                (FilingEvidence("S1", "Investment", "The company approved a facility."),),
            )
        )

    assert captured.value.code == "AI_INVALID_OUTPUT"


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        return SimpleNamespace(
            output_parsed=SimpleNamespace(
                what="The company approved a new facility.",
                what_ko="회사가 새 시설을 승인했다.",
                why_ko="생산능력을 확대하기 위해서다.",
                impact_ko="원문에 영향이 명시되지 않았다.",
                why="The filing states that the facility will expand capacity.",
                impact="The facility may increase production capacity.",
                evidence_ids=("S1",),
                sufficient_evidence=True,
                refusal_reason=None,
            )
        )


class HangulResponses:
    async def parse(self, **arguments: object) -> SimpleNamespace:
        del arguments
        return SimpleNamespace(
            output_parsed=SimpleNamespace(
                what="한국항공우주 approved a facility.",
                what_ko="회사가 새 시설을 승인했다.",
                why_ko="생산능력을 확대하기 위해서다.",
                impact_ko="원문에 영향이 명시되지 않았다.",
                why="The filing states the reason.",
                impact="The facility may increase capacity.",
                evidence_ids=("S1",),
                sufficient_evidence=True,
                refusal_reason=None,
            )
        )
