from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Request
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.errors import AppError
from k_market_ai.rag.application.ask_disclosure import AskDisclosureHandler
from k_market_ai.rag.domain.models import SelectedContext

router = APIRouter(prefix="/internal/v1/disclosures", tags=["disclosure-rag"])


class SelectedContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: UUID
    text: str = Field(min_length=1, max_length=2_000)


class DisclosureQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    selected_context: SelectedContextRequest | None = None


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
        )
    )
    answer = await handler.ask(receipt_number, body.question, selected)
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
