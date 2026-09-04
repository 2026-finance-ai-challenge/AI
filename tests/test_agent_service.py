import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APITimeoutError

from k_market_ai.agent.service import AgentEvidence, AgentHistoryMessage, MarketAgentService
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
        SimpleNamespace(responses=SimpleNamespace(parse=parse)), Settings()
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

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        return SimpleNamespace(
            output_parsed=SimpleNamespace(
                answer="The latest supplied quote is KRW 78,000. [E1]",
                evidence_ids=("E1",),
                insufficient_evidence=False,
                refusal_reason=None,
                suggested_room_name="Samsung latest quote",
                disclaimer="For information only; verify trading status with your broker.",
                confidence=0.95,
            )
        )
