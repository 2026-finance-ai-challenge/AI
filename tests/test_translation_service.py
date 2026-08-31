import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, OpenAIError, RateLimitError

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.translations.domain import TitleSource
from k_market_ai.translations.service import (
    TranslationService,
    canonical_disclosure_section,
    canonical_news_source,
)


def test_title_batch_validates_hashes_and_restores_input_order() -> None:
    first = _title("T1", "삼성전자 신제품 공개")
    second = _title("T2", "유상증자 결정")
    parsed = SimpleNamespace(
        items=(
            SimpleNamespace(
                id="T2",
                source_hash=second.source_hash,
                translated_text="Decision on Capital Increase with Consideration",
            ),
            SimpleNamespace(
                id="T1",
                source_hash=first.source_hash,
                translated_text="Samsung Electronics Unveils New Product",
            ),
        )
    )
    responses = FakeResponses(parsed)
    service = _service(responses)

    result = asyncio.run(service.translate_titles((first, second), "en", "title-v1"))

    assert [item.id for item in result.items] == ["T1", "T2"]
    assert result.items[0].translated_text == "Samsung Electronics Unveils New Product"
    assert responses.arguments["store"] is False
    assert responses.arguments["timeout"] == 90.0


def test_title_batch_rejects_missing_or_extra_provider_items() -> None:
    source = _title("T1", "공시 제목")
    responses = FakeResponses(SimpleNamespace(items=()))

    with pytest.raises(AppError) as captured:
        asyncio.run(_service(responses).translate_titles((source,), "en", "title-v1"))

    assert captured.value.code == "AI_INVALID_OUTPUT"


def test_english_title_batch_rejects_hangul_in_provider_output() -> None:
    source = _title("T1", "마더스제약 상장예비심사 신청")
    responses = FakeResponses(
        SimpleNamespace(
            items=(
                SimpleNamespace(
                    id="T1",
                    source_hash=source.source_hash,
                    translated_text="마더스제약 Files for KOSDAQ Listing Review",
                ),
            )
        )
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(_service(responses).translate_titles((source,), "en-US", "title-v1"))

    assert captured.value.code == "AI_INVALID_OUTPUT"


def test_english_title_batch_requires_standard_krw_conversion() -> None:
    source = _title("T1", "목표가 240만원, 투자유치 111억원")
    responses = FakeResponses(
        SimpleNamespace(
            items=(
                SimpleNamespace(
                    id="T1",
                    source_hash=source.source_hash,
                    translated_text=(
                        "Target Price at KRW 2.4 million After Raising KRW 11.1 billion"
                    ),
                ),
            )
        )
    )

    result = asyncio.run(_service(responses).translate_titles((source,), "en", "title-v1"))

    payload = json.loads(str(responses.arguments["input"]))
    assert payload["items"][0]["required_currency_conversions"] == [
        {"source_text": "240만원", "english_text": "KRW 2.4 million"},
        {"source_text": "111억원", "english_text": "KRW 11.1 billion"},
    ]
    assert result.items[0].translated_text.startswith("Target Price")


def test_english_title_batch_rejects_romanized_or_missing_currency_conversion() -> None:
    source = _title("T1", "투자유치 344억")
    responses = FakeResponses(
        SimpleNamespace(
            items=(
                SimpleNamespace(
                    id="T1",
                    source_hash=source.source_hash,
                    translated_text="Raises 344 eok won in funding",
                ),
            )
        )
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(_service(responses).translate_titles((source,), "en", "title-v1"))

    assert captured.value.code == "AI_INVALID_OUTPUT"


def test_title_batch_classifies_provider_timeout() -> None:
    source = _title("T1", "공시 제목")
    timeout = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))

    with pytest.raises(AppError) as captured:
        asyncio.run(
            _service(FailingResponses(timeout)).translate_titles((source,), "en", "title-v1")
        )

    assert captured.value.code == "AI_PROVIDER_TIMEOUT"
    assert captured.value.status_code == 504


def test_title_batch_classifies_exhausted_quota() -> None:
    source = _title("T1", "공시 제목")
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    exhausted = RateLimitError(
        "quota exhausted",
        response=response,
        body={"code": "credit_balance_exhausted"},
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            _service(FailingResponses(exhausted)).translate_titles((source,), "en", "title-v1")
        )

    assert captured.value.code == "AI_PROVIDER_QUOTA_EXHAUSTED"
    assert captured.value.status_code == 503


def test_news_narrative_preserves_paragraph_count_and_source_hash() -> None:
    title = "실적 발표"
    paragraphs = ("매출이 증가했다.", "회사는 해외 수요를 언급했다.")
    source_hash = _hash(canonical_news_source(title, paragraphs, "SOURCE_EXCERPT"))
    responses = FakeResponses(
        SimpleNamespace(
            translated_paragraphs=(
                "Revenue increased.",
                "The company cited overseas demand.",
            ),
            what="Revenue increased.",
            why="The company cited overseas demand.",
            impact="The excerpt does not state an impact.",
        )
    )

    result = asyncio.run(
        _service(responses).translate_news_narrative(
            source_hash,
            title,
            paragraphs,
            "SOURCE_EXCERPT",
            "en",
            "news-v1",
        )
    )

    assert result.content_availability == "SOURCE_EXCERPT"
    assert len(result.translated_paragraphs) == len(paragraphs)
    assert responses.arguments["timeout"] == 60.0


def test_long_news_narrative_translates_in_bounded_chunks_and_summarizes_once() -> None:
    title = "장문 기사"
    paragraphs = ("가" * 10_000, "나" * 10_000, "다" * 2_000)
    source_hash = _hash(canonical_news_source(title, paragraphs, "FULL_ARTICLE"))
    responses = ChunkedNewsResponses()

    result = asyncio.run(
        _service(responses).translate_news_narrative(
            source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v2"
        )
    )

    assert result.translated_paragraphs == tuple(f"EN:{item}" for item in paragraphs)
    assert responses.summary_calls == 1
    assert responses.translation_calls == 2


def test_disclosure_section_rejects_changed_table_structure() -> None:
    table = json.dumps({"rows": [["매출", 100]]}, ensure_ascii=False)
    source_hash = _hash(canonical_disclosure_section("재무 정보", "매출 현황", table))
    responses = FakeResponses(
        SimpleNamespace(
            translated_heading="Financial Information",
            translated_text="Revenue status",
            translated_table_data_json=json.dumps({"rows": [["Revenue", 101]]}),
        )
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            _service(responses).translate_disclosure_section(
                source_hash,
                "재무 정보",
                "매출 현황",
                table,
                "en",
                "section-v1",
            )
        )

    assert captured.value.code == "AI_INVALID_OUTPUT"


def test_disclosure_section_preserves_table_keys_and_non_string_values() -> None:
    table = json.dumps({"rows": [["매출", 100, True, None]]}, ensure_ascii=False)
    source_hash = _hash(canonical_disclosure_section("재무 정보", "매출 현황", table))
    translated_table = json.dumps({"rows": [["Revenue", 100, True, None]]})
    responses = FakeResponses(
        SimpleNamespace(
            translated_heading="Financial Information",
            translated_text="Revenue status",
            translated_table_data_json=translated_table,
        )
    )

    result = asyncio.run(
        _service(responses).translate_disclosure_section(
            source_hash,
            "재무 정보",
            "매출 현황",
            table,
            "en",
            "section-v1",
        )
    )

    assert result.translated_table_data_json == translated_table
    assert responses.arguments["timeout"] == 90.0


class FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.arguments: dict[str, object] = {}

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        return SimpleNamespace(output_parsed=self.parsed)


class FailingResponses:
    def __init__(self, exception: OpenAIError) -> None:
        self.exception = exception

    async def parse(self, **arguments: object) -> SimpleNamespace:
        del arguments
        raise self.exception


class ChunkedNewsResponses:
    def __init__(self) -> None:
        self.translation_calls = 0
        self.summary_calls = 0

    async def parse(self, **arguments: object) -> SimpleNamespace:
        payload = json.loads(str(arguments["input"]))
        if "source_excerpt" in payload:
            self.summary_calls += 1
            parsed = SimpleNamespace(
                what="The company announced an update.",
                why="The source states the reason.",
                impact="The source describes a potential impact.",
            )
        else:
            self.translation_calls += 1
            parsed = SimpleNamespace(
                translated_paragraphs=tuple(
                    f"EN:{paragraph}" for paragraph in payload["source_paragraphs"]
                )
            )
        return SimpleNamespace(output_parsed=parsed)


def _service(
    responses: FakeResponses | FailingResponses | ChunkedNewsResponses,
) -> TranslationService:
    return TranslationService(
        SimpleNamespace(responses=responses),
        Settings(environment="test", translation_model="translation-test-model"),
    )


def _title(identifier: str, source_text: str) -> TitleSource:
    return TitleSource(identifier, _hash(source_text), source_text)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
