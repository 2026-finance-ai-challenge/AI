import asyncio
from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from k_market_ai.rag.application.ask_disclosure import AskDisclosureHandler
from k_market_ai.rag.domain.models import (
    EmbeddedChunk,
    GeneratedAnswer,
    IndexJob,
    SearchHit,
    SelectedContext,
    SourceSection,
)


@pytest.mark.parametrize("locale", ["en", "ko"])
def test_grounded_answer_keeps_verified_citation(locale) -> None:
    hit = _hit(score=0.8)
    repository = FakeRepository(hits=[hit])
    handler = AskDisclosureHandler(
        repository,
        FakeEmbedding(),
        FakeAnswer(
            GeneratedAnswer(
                answer="Revenue increased during the period. [C1]",
                sufficient_evidence=True,
                citation_ids=("C1",),
                refusal_reason=None,
                model="test-model",
            )
        ),
    )

    answer = asyncio.run(handler.ask("20260818800670", "What changed?", None, locale))

    assert answer.refused is False
    assert answer.citations[0].chunk_id == hit.chunk_id
    assert answer.citations[0].document_version == 1


@pytest.mark.parametrize("locale", ["en", "ko"])
def test_irrelevant_retrieval_refuses_without_generation(locale) -> None:
    answer_port = FakeAnswer(None)
    handler = AskDisclosureHandler(
        FakeRepository(hits=[_hit(score=0.1)]),
        FakeEmbedding(),
        answer_port,
    )

    answer = asyncio.run(handler.ask("20260818800670", "Unrelated question", None, locale))

    assert answer.refused is True
    assert answer.citations == ()
    assert answer_port.called is False
    assert ("근거" in answer.answer) == (locale == "ko")


def test_invalid_selected_text_is_rejected() -> None:
    handler = AskDisclosureHandler(
        FakeRepository(hits=[], selected_text_valid=False),
        FakeEmbedding(),
        FakeAnswer(None),
    )

    try:
        asyncio.run(
            handler.ask(
                "20260818800670",
                "What does this mean?",
                SelectedContext(section_id=uuid4(), text="not in filing"),
            )
        )
    except Exception as exception:
        assert getattr(exception, "code", None) == "INVALID_SELECTED_CONTEXT"
    else:
        raise AssertionError("Expected selected context validation error")


def test_question_language_refusal_uses_current_question_without_invented_evidence() -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    generate = AsyncMock(
        return_value=GeneratedAnswer("", False, (), "관련 근거가 없습니다.", "test-model", "ko")
    )
    handler = AskDisclosureHandler(
        FakeRepository(hits=[]), FakeEmbedding(), SimpleNamespace(answer=generate)
    )
    answer = asyncio.run(handler.ask("20260818800670", "배당 계획은 있나요?", None))
    assert answer.refused is True
    assert "근거" in answer.answer
    assert answer.citations == ()
    generate.assert_not_awaited()


class FakeEmbedding:
    model = "test-embedding"
    dimensions = 3

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(0.1, 0.2, 0.3) for _ in texts]


class FakeAnswer:
    def __init__(self, result: GeneratedAnswer | None) -> None:
        self._result = result
        self.called = False

    async def answer(
        self,
        question: str,
        contexts: Sequence[tuple[str, SearchHit]],
        answer_locale: str = "en",
    ) -> GeneratedAnswer:
        self.called = True
        assert question
        assert contexts
        if self._result is None:
            raise AssertionError("Answer generation must not be called")
        return self._result


class FakeRepository:
    def __init__(self, hits: list[SearchHit], selected_text_valid: bool = True) -> None:
        self._hits = hits
        self._selected_text_valid = selected_text_valid

    async def claim_index_job(self, worker_id: str) -> IndexJob | None:
        return None

    async def load_current_sections(self, receipt_number: str) -> list[SourceSection]:
        return []

    async def complete_index_job(
        self,
        receipt_number: str,
        chunks: Sequence[EmbeddedChunk],
        embedding_model: str,
        embedding_dimensions: int,
        chunker_version: str,
    ) -> None:
        return None

    async def retry_index_job(
        self,
        receipt_number: str,
        error_code: str,
        delay: timedelta,
    ) -> None:
        return None

    async def fail_index_job(self, receipt_number: str, error_code: str) -> None:
        return None

    async def selected_text_exists(
        self,
        receipt_number: str,
        section_id: UUID,
        normalized_text: str,
        translation_source_hash: str | None = None,
    ) -> bool:
        return self._selected_text_valid

    async def search(
        self,
        receipt_number: str,
        embedding: Sequence[float],
        embedding_model: str,
        selected_section_id: UUID | None,
        limit: int,
    ) -> list[SearchHit]:
        return self._hits[:limit]


def _hit(score: float) -> SearchHit:
    section_id = uuid4()
    return SearchHit(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version=1,
        section_ids=(section_id,),
        first_ordinal=3,
        last_ordinal=4,
        heading="Performance",
        content="Revenue increased during the period.",
        score=score,
        selected_priority=2,
    )
