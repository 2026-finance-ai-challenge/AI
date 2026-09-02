import re

from k_market_ai.core.errors import AppError
from k_market_ai.rag.application.ports import AnswerPort, EmbeddingPort, RagRepository
from k_market_ai.rag.domain.chunker import normalize_text
from k_market_ai.rag.domain.errors import RagProviderError
from k_market_ai.rag.domain.models import (
    Citation,
    RagAnswer,
    SearchHit,
    SelectedContext,
)

PROMPT_VERSION = "filing-grounded-v5"
MIN_RELEVANCE = 0.28
SEARCH_LIMIT = 6
REFUSAL_MESSAGE = "I could not find sufficient evidence in this filing to answer that question."


class AskDisclosureHandler:
    def __init__(
        self,
        repository: RagRepository,
        embedding: EmbeddingPort,
        answer: AnswerPort,
    ) -> None:
        self._repository = repository
        self._embedding = embedding
        self._answer = answer

    async def ask(
        self,
        receipt_number: str,
        question: str,
        selected: SelectedContext | None,
    ) -> RagAnswer:
        selected_text = await self._validate_selected(receipt_number, selected)
        query = question if selected_text is None else f"{question}\nSelected text: {selected_text}"
        vector = (await self._embedding.embed([query]))[0]
        hits = await self._repository.search(
            receipt_number,
            vector,
            self._embedding.model,
            selected.section_id if selected else None,
            SEARCH_LIMIT,
        )
        relevant = [hit for hit in hits if hit.selected_priority <= 1 or hit.score >= MIN_RELEVANCE]
        if not relevant:
            return _refusal()

        contexts = [(f"C{index}", hit) for index, hit in enumerate(relevant, start=1)]
        try:
            generated = await self._answer.answer(question, contexts)
        except RagProviderError as exception:
            raise AppError(
                code="MODEL_UNAVAILABLE",
                message="The AI service is temporarily unavailable.",
                status_code=503,
            ) from exception

        if not generated.sufficient_evidence:
            return _refusal(generated.refusal_reason, generated.model)

        available = {context_id: hit for context_id, hit in contexts}
        cited_ids = tuple(dict.fromkeys(generated.citation_ids))
        if not cited_ids or any(citation_id not in available for citation_id in cited_ids):
            return _refusal(
                "The generated answer did not contain valid filing citations.",
                generated.model,
            )
        markers = set(re.findall(r"\[(C[0-9]+)]", generated.answer))
        if not set(cited_ids).issubset(markers):
            return _refusal(
                "The generated answer did not link its claims to filing citations.",
                generated.model,
            )

        citations = tuple(
            _citation(citation_id, available[citation_id]) for citation_id in cited_ids
        )
        return RagAnswer(
            answer=generated.answer,
            refused=False,
            refusal_reason=None,
            citations=citations,
            model=generated.model,
            prompt_version=PROMPT_VERSION,
        )

    async def _validate_selected(
        self,
        receipt_number: str,
        selected: SelectedContext | None,
    ) -> str | None:
        if selected is None:
            return None
        normalized = normalize_text(selected.text)
        if not normalized or not await self._repository.selected_text_exists(
            receipt_number,
            selected.section_id,
            normalized,
        ):
            raise AppError(
                code="INVALID_SELECTED_CONTEXT",
                message="The selected text does not belong to the current filing.",
                status_code=400,
            )
        return normalized


def _citation(citation_id: str, hit: SearchHit) -> Citation:
    excerpt = hit.content if len(hit.content) <= 500 else f"{hit.content[:497]}..."
    return Citation(
        id=citation_id,
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        document_version=hit.document_version,
        section_ids=hit.section_ids,
        first_ordinal=hit.first_ordinal,
        last_ordinal=hit.last_ordinal,
        heading=hit.heading,
        excerpt=excerpt,
    )


def _refusal(reason: str | None = None, model: str | None = None) -> RagAnswer:
    return RagAnswer(
        answer=REFUSAL_MESSAGE,
        refused=True,
        refusal_reason=reason or "No sufficiently relevant filing evidence was retrieved.",
        citations=(),
        model=model,
        prompt_version=PROMPT_VERSION,
    )
