from datetime import date
from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Request
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.answer_language import AnswerLocale
from k_market_ai.core.errors import AppError
from k_market_ai.rag.application.ask_disclosure import AskDisclosureHandler
from k_market_ai.rag.domain.models import SelectedContext

router = APIRouter(prefix="/internal/v1/disclosures", tags=["disclosure-rag"])


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stock_codes: list[Annotated[str, Field(pattern=r"^[A-Z0-9]{6}$")]] = Field(
        min_length=1, max_length=2
    )
    question: str = Field(min_length=1, max_length=4000)
    from_date: date | None = None
    to_date: date | None = None
    financials: bool = False


@router.post("/evidence")
async def retrieve_evidence(
    request: Request,
    body: EvidenceRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> list[dict[str, object]]:
    evidence = await _handler(request).retrieve(
        body.stock_codes,
        body.question,
        body.from_date,
        body.to_date,
        body.financials,
    )
    return [
        {
            "receipt_number": item.filing.receipt_number,
            "stock_code": item.filing.stock_code,
            "title": item.filing.title,
            "filed_date": item.filing.filed_date,
            "detected_at": item.filing.detected_at,
            "content": item.content,
            "section_ids": item.section_ids,
            "retrieval_method": item.retrieval_method,
        }
        for item in evidence
    ]


class SelectedContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: UUID
    text: str = Field(min_length=1, max_length=2_000)
    translation_source_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class DisclosureQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    selected_context: SelectedContextRequest | None = None
    answer_locale: AnswerLocale = "auto"


class CitationResponse(BaseModel):
    id: str
    chunk_id: UUID
    document_id: UUID
    document_version: int
    section_ids: tuple[UUID, ...]
    first_ordinal: int
    last_ordinal: int
    heading: str | None
    excerpt: str


class DisclosureAnswerResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: tuple[CitationResponse, ...]
    model: str | None
    prompt_version: str


@router.post(
    "/{receipt_number}/questions",
    response_model=DisclosureAnswerResponse,
)
async def ask_disclosure(
    request: Request,
    body: DisclosureQuestionRequest,
    receipt_number: Annotated[str, Path(pattern=r"^[0-9]{14}$")],
    _: Annotated[None, Depends(authenticate_internal)],
) -> DisclosureAnswerResponse:
    handler = _handler(request)
    selected = (
        None
        if body.selected_context is None
        else SelectedContext(
            section_id=body.selected_context.section_id,
            text=body.selected_context.text,
            translation_source_hash=body.selected_context.translation_source_hash,
        )
    )
    answer = await handler.ask(receipt_number, body.question, selected, body.answer_locale)
    return DisclosureAnswerResponse(
        answer=answer.answer,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        citations=tuple(
            CitationResponse(
                id=citation.id,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                document_version=citation.document_version,
                section_ids=citation.section_ids,
                first_ordinal=citation.first_ordinal,
                last_ordinal=citation.last_ordinal,
                heading=citation.heading,
                excerpt=citation.excerpt,
            )
            for citation in answer.citations
        ),
        model=answer.model,
        prompt_version=answer.prompt_version,
    )


def _handler(request: Request) -> AskDisclosureHandler:
    handler: object | None = getattr(request.app.state, "rag_handler", None)
    if handler is None:
        raise AppError(
            code="RAG_NOT_CONFIGURED",
            message="The RAG service is not configured.",
            status_code=503,
        )
    return cast(AskDisclosureHandler, handler)
