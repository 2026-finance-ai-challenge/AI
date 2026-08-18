import asyncio
from types import SimpleNamespace
from uuid import uuid4

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

    answer = asyncio.run(adapter.answer("What changed?", [("C1", hit)]))

    assert answer.model == ANSWER_MODEL
    assert answer.citation_ids == ("C1",)
    assert responses.arguments["store"] is False
    assert "Revenue increased by 10%." in str(responses.arguments["input"])
    assert "conversation" not in str(responses.arguments["input"]).lower()


class FakeResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        parsed = SimpleNamespace(
            answer="Revenue increased by 10%. [C1]",
            sufficient_evidence=True,
            citation_ids=("C1",),
            refusal_reason=None,
        )
        return SimpleNamespace(output_parsed=parsed)
