import hashlib
from uuid import uuid4

from fastapi.testclient import TestClient

from k_market_ai.core.config import Settings
from k_market_ai.main import create_app
from k_market_ai.translations.domain import (
    DisclosureSectionTranslation,
    NewsNarrative,
    TitleTranslation,
    TitleTranslationBatch,
)
from k_market_ai.translations.service import (
    canonical_disclosure_section,
    canonical_news_source,
)


def test_translation_endpoints_require_token_and_return_bound_metadata() -> None:
    token = str(uuid4())
    service = FakeTranslationService()
    app = create_app(
        Settings(environment="test", allowed_hosts=["testserver"], service_token=token),
        translation_service=service,
    )
    title_hash = _hash("공시 제목")
    news_hash = _hash(canonical_news_source("제목", ("본문",), "SOURCE_EXCERPT"))
    section_hash = _hash(canonical_disclosure_section("제목", "본문", None))
    headers = {"authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        unauthorized = client.post(
            "/internal/v1/translations/titles",
            json={
                "items": [{"id": "T1", "source_hash": title_hash, "source_text": "공시 제목"}],
                "translation_version": "title-v1",
            },
        )
        titles = client.post(
            "/internal/v1/translations/titles",
            headers=headers,
            json={
                "items": [{"id": "T1", "source_hash": title_hash, "source_text": "공시 제목"}],
                "translation_version": "title-v1",
            },
        )
        narrative = client.post(
            "/internal/v1/news/narratives",
            headers=headers,
            json={
                "source_hash": news_hash,
                "title": "제목",
                "paragraphs": ["본문"],
                "content_availability": "SOURCE_EXCERPT",
                "translation_version": "news-v1",
            },
        )
        section_id = uuid4()
        section = client.post(
            "/internal/v1/disclosures/section-translations",
            headers=headers,
            json={
                "receipt_number": "20260823800001",
                "document_version": 3,
                "section_id": str(section_id),
                "source_hash": section_hash,
                "heading": "제목",
                "text": "본문",
                "translation_version": "section-v1",
            },
        )

    assert unauthorized.status_code == 401
    assert titles.status_code == 200
    assert titles.json()["items"][0]["source_hash"] == title_hash
    assert narrative.status_code == 200
    assert narrative.json()["content_availability"] == "SOURCE_EXCERPT"
    assert section.status_code == 200
    assert section.json()["section_id"] == str(section_id)
    assert section.json()["document_version"] == 3


class FakeTranslationService:
    async def translate_titles(
        self, items: tuple[object, ...], target_locale: str, translation_version: str
    ) -> TitleTranslationBatch:
        item = items[0]
        return TitleTranslationBatch(
            (TitleTranslation(item.id, item.source_hash, "Filing Title"),),
            target_locale,
            translation_version,
            "test-model",
            "title-test-v1",
        )

    async def translate_news_narrative(
        self,
        source_hash: str,
        title: str,
        paragraphs: tuple[str, ...],
        content_availability: str,
        target_locale: str,
        translation_version: str,
    ) -> NewsNarrative:
        del title, paragraphs
        return NewsNarrative(
            source_hash,
            ("Body",),
            "What",
            "Why",
            "Impact",
            content_availability,
            target_locale,
            translation_version,
            "test-model",
            "news-test-v1",
        )

    async def translate_disclosure_section(
        self,
        source_hash: str,
        heading: str | None,
        text: str | None,
        table_data_json: str | None,
        target_locale: str,
        translation_version: str,
    ) -> DisclosureSectionTranslation:
        del heading, text, table_data_json
        return DisclosureSectionTranslation(
            source_hash,
            "Heading",
            "Body",
            None,
            target_locale,
            translation_version,
            "test-model",
            "section-test-v1",
        )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
