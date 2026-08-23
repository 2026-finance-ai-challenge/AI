from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.errors import AppError
from k_market_ai.news.domain import TermEvidence
from k_market_ai.news.service import NewsIntelligenceService

router = APIRouter(prefix="/internal/v1/news", tags=["news-intelligence"])


class NewsAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=1_000)
    paragraphs: tuple[str, ...] = Field(min_length=1, max_length=200)
    candidate_companies: tuple[str, ...] = Field(default=(), max_length=75)


class NewsAnalysisResponse(BaseModel):
    english_title: str
    translated_paragraphs: tuple[str, ...]
    what: str
    why: str
    impact: str
    event_type: str
    sentiment: str
    importance: str
    market_impact: str
    market_impact_importance: str
    market_impact_score: float = Field(ge=0, le=1)
    event_confidence: float
    sentiment_confidence: float
    importance_confidence: float
    market_impact_confidence: float
    model: str
    prompt_version: str


class TermEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{0,31}$")
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=3_000)
    source_url: str | None = Field(default=None, max_length=2_000)


class TermExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_text: str = Field(min_length=1, max_length=500)
    article_context: str = Field(min_length=1, max_length=6_000)
    evidence: tuple[TermEvidenceRequest, ...] = Field(default=(), max_length=8)
    safety_identifier: str | None = Field(default=None, pattern=r"^[a-f0-9]{32,64}$")


class TermExplanationResponse(BaseModel):
    normalized_term: str | None
    definition: str | None
    contextual_meaning: str | None
    evidence_ids: tuple[str, ...]
    confidence: float
    review_required: bool
    sufficient_evidence: bool
    refusal_reason: str | None
    model: str
    prompt_version: str


@router.post("/analysis", response_model=NewsAnalysisResponse)
async def analyze_news(
    request: Request,
    body: NewsAnalysisRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> NewsAnalysisResponse:
    result = await _service(request).analyze(
        body.title,
        body.paragraphs,
        body.candidate_companies,
    )
    return NewsAnalysisResponse(
        english_title=result.english_title,
        translated_paragraphs=result.translated_paragraphs,
        what=result.what,
        why=result.why,
        impact=result.impact,
        event_type=result.event_type,
        sentiment=result.sentiment,
        importance=result.importance,
        market_impact=result.market_impact,
        market_impact_importance=result.market_impact_importance,
        market_impact_score=result.market_impact_score,
        event_confidence=result.event_confidence,
        sentiment_confidence=result.sentiment_confidence,
        importance_confidence=result.importance_confidence,
        market_impact_confidence=result.market_impact_confidence,
        model=result.model,
        prompt_version=result.prompt_version,
    )


@router.post("/terms/explanations", response_model=TermExplanationResponse)
async def explain_term(
    request: Request,
    body: TermExplanationRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> TermExplanationResponse:
    result = await _service(request).explain_term(
        body.selected_text,
        body.article_context,
        tuple(
            TermEvidence(item.id, item.title, item.content, item.source_url)
            for item in body.evidence
        ),
        body.safety_identifier,
    )
    return TermExplanationResponse(
        normalized_term=result.normalized_term,
        definition=result.definition,
        contextual_meaning=result.contextual_meaning,
        evidence_ids=result.evidence_ids,
        confidence=result.confidence,
        review_required=result.review_required,
        sufficient_evidence=result.sufficient_evidence,
        refusal_reason=result.refusal_reason,
        model=result.model,
        prompt_version=result.prompt_version,
    )


def _service(request: Request) -> NewsIntelligenceService:
    service: object | None = getattr(request.app.state, "news_service", None)
    if service is None:
        raise AppError(
            code="NEWS_AI_NOT_CONFIGURED",
            message="The news AI service is not configured.",
            status_code=503,
        )
    return cast(NewsIntelligenceService, service)
