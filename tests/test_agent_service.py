import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APITimeoutError

from k_market_ai.agent.service import (
    AgentEvidence,
    AgentHistoryMessage,
    MarketAgentService,
    _generation_schema,
)
from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError


def test_market_agent_uses_server_evidence_without_provider_storage() -> None:
    responses = FakeResponses()
    service = MarketAgentService(
        SimpleNamespace(responses=responses),
        Settings(
            environment="test",
            agent_model="test-agent-model",
            agent_prompt_version="market-agent-test-v2",
        ),
    )

    result = asyncio.run(
        service.answer(
            "STOCK",
            "Samsung Electronics",
            "What is the latest observed price?",
            (
                AgentHistoryMessage("USER", "Tell me about this company."),
                AgentHistoryMessage("ASSISTANT", "Earlier answer."),
            ),
            (
                AgentEvidence(
                    "E1",
                    "Observed quote",
                    "The observed price is KRW 78,000.",
                    "KIS",
                    "2026-08-23T01:00:00Z",
                ),
            ),
            "a" * 64,
            "en",
        )
    )

    assert result.answer == "The latest supplied quote is KRW 78,000. [E1]"
    assert result.evidence_ids == ("E1",)
    assert result.prompt_version == "market-agent-test-v2"
    assert responses.arguments["store"] is False
    assert responses.arguments["safety_identifier"] == "a" * 64
    assert responses.arguments["timeout"] == 90.0
    assert responses.arguments["reasoning"] == {"effort": "medium"}
    assert responses.arguments["max_output_tokens"] == 16_000
    wire = responses.arguments["text"]
    assert wire["verbosity"] == "low"
    assert wire["format"]["strict"] is True
    assert wire["format"]["schema"]["additionalProperties"] is False
    assert "maxLength" not in wire["format"]["schema"]["properties"]["answer"]
    assert "refusal_reason" in wire["format"]["schema"]["required"]
    assert "KRW 78,000" in str(responses.arguments["input"])
    instructions = str(responses.arguments["instructions"])
    assert "Answer the current question" in instructions
    assert "insufficient_evidence refers only to the current question" in instructions
    assert "exact reporting-period column" in instructions
    messages = responses.arguments["input"]
    assert messages[0] == {"role": "user", "content": "Tell me about this company."}
    assert messages[1] == {"role": "assistant", "content": "Earlier answer."}
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("Current question:\nWhat is the latest observed price?")


@pytest.mark.parametrize("deadline", [False, True])
def test_market_agent_timeout_is_distinct_and_never_retries(deadline: bool) -> None:
    failure = (
        TimeoutError()
        if deadline
        else APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    )
    parse = AsyncMock(side_effect=failure)
    service = MarketAgentService(
        SimpleNamespace(responses=SimpleNamespace(create=parse)), Settings()
    )
    with pytest.raises(AppError) as caught:
        asyncio.run(service.answer("NEWS", "News", "Explain this.", (), (), "a" * 64))
    assert caught.value.code == "AI_PROVIDER_TIMEOUT"
    parse.assert_awaited_once()


def test_market_agent_timeout_cannot_outlive_backend_request() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(agent_timeout_seconds=120)


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def create(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        return SimpleNamespace(
            status="completed",
            output=[],
            output_text=json.dumps(
                dict(
                    answer="The latest supplied quote is KRW 78,000. [E1]",
                    evidence_ids=("E1",),
                    insufficient_evidence=False,
                    refusal_reason=None,
                    suggested_room_name="Samsung latest quote",
                    disclaimer="For information only; verify trading status with your broker.",
                    confidence=0.95,
                )
            ),
        )


def envelope(status="completed", text='{"answer":"unfinished', reason=None):
    return SimpleNamespace(
        status=status,
        output_text=text,
        output=[],
        id="response-test",
        model="gpt-5-nano",
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
        usage=SimpleNamespace(
            output_tokens=16000, output_tokens_details=SimpleNamespace(reasoning_tokens=15000)
        ),
        max_output_tokens=16000,
    )


@pytest.mark.parametrize(
    "status,reason",
    [("incomplete", "max_output_tokens"), ("incomplete", "content_filter"), ("failed", None)],
)
def test_incomplete_envelope_is_rejected_before_json_parsing_without_retry(status, reason, caplog):
    request = AsyncMock(return_value=envelope(status, '{"answer":"PRIVATE_OUTPUT', reason))
    service = MarketAgentService(
        SimpleNamespace(responses=SimpleNamespace(create=request)), Settings()
    )
    with pytest.raises(AppError) as caught:
        asyncio.run(
            service.answer("GENERAL", "PRIVATE_TITLE", "PRIVATE_QUESTION", (), (), "a" * 64)
        )
    assert caught.value.code == "AI_GENERATION_INCOMPLETE"
    request.assert_awaited_once()
    assert "output_tokens=16000" in caplog.text
    assert "reasoning_tokens=15000" in caplog.text
    assert "response-test" in caplog.text
    assert "PRIVATE_" not in caplog.text
    assert "출력 스키마 실패" not in caplog.text


def test_completed_but_malformed_json_is_not_repaired_or_saved(caplog):
    request = AsyncMock(return_value=envelope())
    service = MarketAgentService(
        SimpleNamespace(responses=SimpleNamespace(create=request)), Settings()
    )
    with pytest.raises(AppError) as caught:
        asyncio.run(service.answer("GENERAL", "Market", "Kakao earnings", (), (), "a" * 64))
    assert caught.value.code == "AI_INVALID_OUTPUT"
    assert "schema_validation" in caplog.text and "json_invalid" in caplog.text
    request.assert_awaited_once()


def test_provider_refusal_is_not_parsed_or_logged_as_answer(caplog):
    response = envelope(text="")
    response.output = [
        SimpleNamespace(
            type="message", content=[SimpleNamespace(type="refusal", refusal="PRIVATE_REFUSAL")]
        )
    ]
    request = AsyncMock(return_value=response)
    service = MarketAgentService(
        SimpleNamespace(responses=SimpleNamespace(create=request)), Settings()
    )
    with pytest.raises(AppError) as caught:
        asyncio.run(service.answer("GENERAL", "Market", "Question", (), (), "a" * 64))
    assert caught.value.code == "AI_PROVIDER_REFUSAL"
    assert "PRIVATE_REFUSAL" not in caplog.text
    request.assert_awaited_once()


@pytest.mark.parametrize(
    "field,value", [("answer", "x" * 10001), ("confidence", 1.1), ("answer", "한글")]
)
def test_generation_schema_does_not_weaken_final_validation(field, value):
    data = dict(
        answer="Kakao earnings.",
        evidence_ids=[],
        insufficient_evidence=False,
        refusal_reason=None,
        suggested_room_name="Kakao",
        disclaimer="Information only.",
        confidence=0.5,
    )
    data[field] = value
    request = AsyncMock(return_value=envelope(text=json.dumps(data)))
    service = MarketAgentService(
        SimpleNamespace(responses=SimpleNamespace(create=request)), Settings()
    )
    with pytest.raises(AppError) as caught:
        asyncio.run(service.answer("GENERAL", "Market", "Kakao earnings", (), (), "a" * 64, "en"))
    assert caught.value.code == "AI_INVALID_OUTPUT"
    request.assert_awaited_once()


@pytest.mark.parametrize(
    "text,accepted",
    [
        ("Kakao’s revenue was ₩4.04 trillion.\nNet income was positive.", True),
        ("Kakao's net income (당기순이익) rose.", False),
        ("Kakao profit 純利益 rose.", False),
        ("Kakao カカオ earnings.", False),
        ("Kakao ᄏ earnings.", False),
    ],
)
def test_generation_grammar_rejects_mixed_scripts_without_rejecting_english_symbols(text, accepted):
    pattern = _generation_schema("en")["properties"]["answer"]["pattern"]
    assert bool(re.search(pattern, text)) == accepted
