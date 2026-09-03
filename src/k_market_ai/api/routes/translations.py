import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.errors import AppError
from k_market_ai.translations.domain import TitleSource
from k_market_ai.translations.service import TranslationService

router = APIRouter(tags=["translations"])
GENERATION_DEADLINE_SECONDS = 180


@asynccontextmanager
async def generation_deadline() -> AsyncIterator[None]:
    try:
        # 세마포어 대기와 모든 분할 요청을 포함해 백엔드 작업 임대 안에서 끝낸다.
        async with asyncio.timeout(GENERATION_DEADLINE_SECONDS):
            yield
    except TimeoutError as exception:
        raise AppError(
            code="AI_PROVIDER_TIMEOUT", message="Translation generation timed out.", status_code=503
        ) from exception


BoundedParagraph = Annotated[str, Field(min_length=1, max_length=120_000)]


class TitleSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_text: str = Field(min_length=1, max_length=1_000)


class TitleTranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: tuple[TitleSourceRequest, ...] = Field(min_length=1, max_length=25)
    target_locale: Literal["en"] = "en"
    translation_version: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9._-]+$")


class TitleTranslationItemResponse(BaseModel):
    id: str
    source_hash: str
    translated_text: str


class TitleTranslationResponse(BaseModel):
    items: tuple[TitleTranslationItemResponse, ...]
    target_locale: str
    translation_version: str
    model: str
    prompt_version: str


class NewsNarrativeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    title: str = Field(min_length=1, max_length=1_000)
    paragraphs: tuple[BoundedParagraph, ...] = Field(min_length=1, max_length=500)
    content_availability: Literal["FULL_ARTICLE", "SOURCE_EXCERPT"]
    target_locale: Literal["en", "ko"] = "en"
    translation_version: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9._-]+$")


class NewsNarrativeResponse(BaseModel):
    source_hash: str
    translated_paragraphs: tuple[str, ...]
    what: str
    why: str
    impact: str
    content_availability: str
    target_locale: str
    translation_version: str
    model: str
    prompt_version: str


class DisclosureSectionTranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_number: str = Field(pattern=r"^[0-9]{14}$")
    document_version: int = Field(ge=1, le=1_000_000)
    section_id: UUID
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    heading: str | None = Field(default=None, max_length=4_000)
    text: str | None = Field(default=None, max_length=120_000)
    table_data_json: str | None = Field(default=None, max_length=500_000)
    target_locale: Literal["en"] = "en"
    translation_version: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9._-]+$")


class DisclosureSectionTranslationResponse(BaseModel):
    receipt_number: str
    document_version: int
    section_id: UUID
    source_hash: str
    translated_heading: str | None
    translated_text: str | None
    translated_table_data_json: str | None
    target_locale: str
    translation_version: str
    model: str
    prompt_version: str


@router.post("/internal/v1/translations/titles", response_model=TitleTranslationResponse)
async def translate_titles(
    request: Request,
    body: TitleTranslationRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> TitleTranslationResponse:
    async with generation_deadline():
        result = await _service(request).translate_titles(
            tuple(TitleSource(item.id, item.source_hash, item.source_text) for item in body.items),
            body.target_locale,
            body.translation_version,
        )
    return TitleTranslationResponse(
        items=tuple(
            TitleTranslationItemResponse(
                id=item.id,
                source_hash=item.source_hash,
                translated_text=item.translated_text,
            )
            for item in result.items
        ),
        target_locale=result.target_locale,
        translation_version=result.translation_version,
        model=result.model,
        prompt_version=result.prompt_version,
    )


@router.post("/internal/v1/news/narratives", response_model=NewsNarrativeResponse)
async def translate_news_narrative(
    request: Request,
    body: NewsNarrativeRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> NewsNarrativeResponse:
    async with generation_deadline():
        result = await _service(request).translate_news_narrative(
            body.source_hash,
            body.title,
            body.paragraphs,
            body.content_availability,
            body.target_locale,
            body.translation_version,
        )
    return NewsNarrativeResponse(
        source_hash=result.source_hash,
        translated_paragraphs=result.translated_paragraphs,
        what=result.what,
        why=result.why,
        impact=result.impact,
        content_availability=result.content_availability,
        target_locale=result.target_locale,
        translation_version=result.translation_version,
        model=result.model,
        prompt_version=result.prompt_version,
    )


@router.post("/internal/v1/news/narratives/stream")
async def stream_news_narrative(
    request: Request,
    body: NewsNarrativeRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> StreamingResponse:
    service = _service(request)

    async def events() -> AsyncIterator[str]:
        try:
            async with generation_deadline():
                async for event in service.stream_news_bundle(
                    body.source_hash,
                    body.title,
                    body.paragraphs,
                    body.content_availability,
                    body.translation_version,
                ):
                    yield json.dumps(event, ensure_ascii=False) + "\n"
        except AppError as exception:
            yield json.dumps({"type": "error", "code": exception.code}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post(
    "/internal/v1/disclosures/section-translations",
    response_model=DisclosureSectionTranslationResponse,
)
async def translate_disclosure_section(
    request: Request,
    body: DisclosureSectionTranslationRequest,
    _: Annotated[None, Depends(authenticate_internal)],
) -> DisclosureSectionTranslationResponse:
    async with generation_deadline():
        result = await _service(request).translate_disclosure_section(
            body.source_hash,
            body.heading,
            body.text,
            body.table_data_json,
            body.target_locale,
            body.translation_version,
        )
    return DisclosureSectionTranslationResponse(
        receipt_number=body.receipt_number,
        document_version=body.document_version,
        section_id=body.section_id,
        source_hash=result.source_hash,
        translated_heading=result.translated_heading,
        translated_text=result.translated_text,
        translated_table_data_json=result.translated_table_data_json,
        target_locale=result.target_locale,
        translation_version=result.translation_version,
        model=result.model,
        prompt_version=result.prompt_version,
    )


def _service(request: Request) -> TranslationService:
    service: object | None = getattr(request.app.state, "translation_service", None)
    if service is None:
        raise AppError(
            code="TRANSLATION_AI_NOT_CONFIGURED",
            message="The translation AI service is not configured.",
            status_code=503,
        )
    return cast(TranslationService, service)
