from typing import Annotated, Literal, cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.agent.service import (
    AgentEvidence,
    AgentHistoryMessage,
    MarketAgentService,
)
from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.answer_language import AnswerLocale
from k_market_ai.core.errors import AppError

router = APIRouter(prefix="/internal/v1/agent", tags=["market-agent"])


class HistoryMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["USER", "ASSISTANT"]
    content: str = Field(min_length=1, max_length=12_000)


class AgentEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^E[0-9]{1,3}$")
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=12_000)
    source: str = Field(min_length=1, max_length=100)
    as_of: str | None = Field(default=None, max_length=64)


class AgentAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context_type: Literal["GENERAL", "STOCK", "NEWS", "TAX_GUIDE"]
    context_title: str = Field(min_length=1, max_length=500)
    question: str = Field(min_length=1, max_length=4_000)
    history: tuple[HistoryMessageRequest, ...] = Field(max_length=20)
    evidence: tuple[AgentEvidenceRequest, ...] = Field(max_length=20)
    safety_identifier: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_locale: AnswerLocale = "auto"


class AgentAnswerResponse(BaseModel):
    answer: str
    evidence_ids: tuple[str, ...]
    insufficient_evidence: bool
    refusal_reason: str | None
    suggested_room_name: str
    disclaimer: str
    confidence: float
    model: str
    prompt_version: str


@router.post("/answers", response_model=AgentAnswerResponse)
async def answer(
    request: Request,
    body: AgentAnswerRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> AgentAnswerResponse:
    result = await _service(request).answer(
        body.context_type,
        body.context_title,
        body.question,
        tuple(AgentHistoryMessage(item.role, item.content) for item in body.history),
        tuple(
            AgentEvidence(item.id, item.title, item.content, item.source, item.as_of)
            for item in body.evidence
        ),
        body.safety_identifier,
        body.answer_locale,
    )
    return AgentAnswerResponse(
        answer=result.answer,
        evidence_ids=result.evidence_ids,
        insufficient_evidence=result.insufficient_evidence,
        refusal_reason=result.refusal_reason,
        suggested_room_name=result.suggested_room_name,
        disclaimer=result.disclaimer,
        confidence=result.confidence,
        model=result.model,
        prompt_version=result.prompt_version,
    )


def _service(request: Request) -> MarketAgentService:
    service: object | None = getattr(request.app.state, "agent_service", None)
    if service is None:
        raise AppError(
            code="MARKET_AGENT_NOT_CONFIGURED",
            message="The market agent service is not configured.",
            status_code=503,
        )
    return cast(MarketAgentService, service)
