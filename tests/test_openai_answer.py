import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from k_market_ai.rag.domain.errors import RagProviderError
from k_market_ai.rag.domain.models import SearchHit
from k_market_ai.rag.infrastructure.openai_answer import (
    ANSWER_MODEL,
    OpenAIAnswerAdapter,
)


def test_answer_disables_storage_and_sends_only_retrieved_context() -> None:
    responses = FakeResponses()
    adapter = OpenAIAnswerAdapter(SimpleNamespace(responses=responses))
    hit = SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version=1,
        section_ids=(uuid4(),),
        first_ordinal=1,
        last_ordinal=1,
        heading="Revenue",
        content="Revenue increased by 10%.",
        score=0.9,
        selected_priority=2,
    )

    answer = asyncio.run(adapter.answer("What changed?", [("C1", hit)], "en"))

    assert answer.model == ANSWER_MODEL
    assert answer.citation_ids == ("C1",)
    assert responses.arguments["store"] is False
    assert "Revenue increased by 10%." in str(responses.arguments["input"])
    assert "conversation" not in str(responses.arguments["input"]).lower()


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def create(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        parsed = SimpleNamespace(
            claims=[{"text": "Revenue increased by 10%.", "citation_ids": ["C1"]}],
            sufficient_evidence=True,
            refusal_reason=None,
        )
        return SimpleNamespace(status="completed", output_text=json.dumps(vars(parsed)))


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_unfinished_provider_response_is_not_parsed_or_retried(status) -> None:
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            status=status,
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            max_output_tokens=8192,
            usage=SimpleNamespace(output_tokens=512),
            output_text='{"answer":',
        )

    adapter = OpenAIAnswerAdapter(SimpleNamespace(responses=SimpleNamespace(create=create)))
    with pytest.raises(RagProviderError, match="did not complete"):
        asyncio.run(adapter.answer("What changed?", [], "en"))
    assert len(calls) == 1


def test_invalid_result_is_rejected_without_logging_source_or_output(caplog) -> None:
    async def create(**kwargs):
        return SimpleNamespace(status="completed", output_text='{"answer":"PRIVATE_OUTPUT"}')

    adapter = OpenAIAnswerAdapter(SimpleNamespace(responses=SimpleNamespace(create=create)))
    with pytest.raises(RagProviderError, match="invalid structured"):
        asyncio.run(adapter.answer("PRIVATE_QUESTION", []))
    assert "PRIVATE_OUTPUT" not in caplog.text
    assert "PRIVATE_QUESTION" not in caplog.text


def test_claim_without_source_is_rejected():
    async def create(**kwargs):
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(
                {
                    "claims": [{"text": "Revenue increased.", "citation_ids": []}],
                    "sufficient_evidence": True,
                    "refusal_reason": None,
                }
            ),
        )

    adapter = OpenAIAnswerAdapter(SimpleNamespace(responses=SimpleNamespace(create=create)))
    with pytest.raises(RagProviderError):
        asyncio.run(adapter.answer("What changed?", []))
