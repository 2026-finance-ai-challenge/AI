import asyncio
from types import SimpleNamespace

from k_market_ai.agent.service import AgentEvidence, AgentHistoryMessage, MarketAgentService
from k_market_ai.core.config import Settings


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
            (AgentHistoryMessage("USER", "Tell me about this company."),),
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
        )
    )

    assert result.answer == "The latest supplied quote is KRW 78,000. [E1]"
    assert result.evidence_ids == ("E1",)
    assert result.prompt_version == "market-agent-test-v2"
    assert responses.arguments["store"] is False
    assert responses.arguments["safety_identifier"] == "a" * 64
    assert "KRW 78,000" in str(responses.arguments["input"])


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
