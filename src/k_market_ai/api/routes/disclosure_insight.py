from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.errors import AppError
from k_market_ai.rag.application.disclosure_insight import (
    DisclosureInsightService,
    FilingEvidence,
)

router = APIRouter(prefix="/internal/v1/disclosures", tags=["disclosure-insight"])


class FilingEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^S[0-9]{1,4}$")
    heading: str | None = Field(default=None, max_length=500)
    content: str = Field(min_length=1, max_length=6_000)


class DisclosureInsightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_number: str = Field(pattern=r"^[0-9]{14}$")
    title: str = Field(min_length=1, max_length=500)
    evidence: tuple[FilingEvidenceRequest, ...] = Field(min_length=1, max_length=100)


class DisclosureInsightResponse(BaseModel):
    what: str | None
    why: str | None
    impact: str | None
    evidence_ids: tuple[str, ...]
    sufficient_evidence: bool
    refusal_reason: str | None
    model: str
    prompt_version: str


@router.post("/summaries", response_model=DisclosureInsightResponse)
async def summarize_disclosure(
    request: Request,
    body: DisclosureInsightRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> DisclosureInsightResponse:
    result = await _service(request).summarize(
        body.receipt_number,
        body.title,
        tuple(FilingEvidence(item.id, item.heading, item.content) for item in body.evidence),
    )
    return DisclosureInsightResponse(
        what=result.what,
        why=result.why,
        impact=result.impact,
        evidence_ids=result.evidence_ids,
        sufficient_evidence=result.sufficient_evidence,
        refusal_reason=result.refusal_reason,
        model=result.model,
        prompt_version=result.prompt_version,
    )


def _service(request: Request) -> DisclosureInsightService:
    service: object | None = getattr(request.app.state, "disclosure_insight_service", None)
    if service is None:
        raise AppError(
            code="DISCLOSURE_INSIGHT_NOT_CONFIGURED",
            message="The disclosure insight service is not configured.",
            status_code=503,
        )
    return cast(DisclosureInsightService, service)
