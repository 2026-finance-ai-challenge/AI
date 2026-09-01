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
    _StructuredNewsSegment,
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
    assert responses.arguments["reasoning"] == {"effort": "minimal"}
    assert responses.arguments["text"] == {"verbosity": "low"}
    assert responses.arguments["store"] is False
    assert responses.arguments["timeout"] == 90.0
    assert responses.arguments["max_output_tokens"] == 128_000


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
                        "Target Price at __KRW_AMOUNT_0__ After Raising __KRW_AMOUNT_1__"
                    ),
                ),
            )
        )
    )

    result = asyncio.run(_service(responses).translate_titles((source,), "en", "title-v1"))

    payload = json.loads(str(responses.arguments["input"]))
    assert payload["items"][0]["source_text"] == (
        "목표가 __KRW_AMOUNT_0__, 투자유치 __KRW_AMOUNT_1__"
    )
    assert payload["items"][0]["protected_currency_tokens"] == [
        "__KRW_AMOUNT_0__",
        "__KRW_AMOUNT_1__",
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


def test_news_narrative_repairs_field_label_placeholders_without_retry() -> None:
    title = "노선 확대"
    paragraphs = ("회사는 가을 수요에 대응해 노선을 확대했다.", "경쟁력을 강화할 계획이다.")
    source_hash = _hash(canonical_news_source(title, paragraphs, "FULL_ARTICLE"))
    responses = FakeResponses(
        SimpleNamespace(
            translated_paragraphs=(
                "The company expanded routes to capture autumn demand.",
                "It plans to strengthen its competitiveness.",
            ),
            what="What",
            why="Why",
            impact="Impact",
        )
    )

    result = asyncio.run(
        _service(responses).translate_news_narrative(
            source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v3"
        )
    )

    assert result.what == "The company expanded routes to capture autumn demand."
    assert result.why == "The company expanded routes to capture autumn demand."
    assert result.impact == "It plans to strengthen its competitiveness."
    assert responses.calls == 3


def test_news_narrative_repairs_non_english_or_unfinished_summaries_without_retry() -> None:
    title = "채용 확대"
    paragraphs = ("회사는 인재 확보를 위해 채용을 확대했다.", "경쟁력을 강화할 계획이다.")
    source_hash = _hash(canonical_news_source(title, paragraphs, "FULL_ARTICLE"))
    responses = FakeResponses(
        SimpleNamespace(
            translated_paragraphs=(
                "The company expanded hiring to secure talent.",
                "It plans to strengthen competitiveness.",
            ),
            what="The company expanded hiring to secure talent.",
            why="The company seeks more talent 高",
            impact="It plans to strengthen competitiveness",
        )
    )

    result = asyncio.run(
        _service(responses).translate_news_narrative(
            source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v6"
        )
    )

    assert result.why == "The source does not state a reason."
    assert result.impact == "It plans to strengthen competitiveness."
    assert responses.calls == 3


def test_news_narrative_enforces_one_short_sentence_without_retry() -> None:
    title = "요약 제한"
    paragraphs = ("회사는 신사업 계획을 발표했다.",)
    source_hash = _hash(canonical_news_source(title, paragraphs, "FULL_ARTICLE"))
    long_summary = (
        "The company announced a detailed new business plan that covers product development "
        "market expansion hiring partnerships financing operations distribution and customer "
        "support across several regions. A second sentence must be removed."
    )
    responses = FakeResponses(
        SimpleNamespace(
            translated_paragraphs=("The company announced a new business plan.",),
            what=long_summary,
            why="The source does not state a reason.",
            impact="The source does not state a direct impact.",
        )
    )

    result = asyncio.run(
        _service(responses).translate_news_narrative(
            source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v6"
        )
    )

    assert len(result.what.split()) <= 24
    assert result.what.endswith((".", "…"))
    assert len(result.what) <= 180
    assert "second sentence" not in result.what.lower()
    assert responses.calls == 2


def test_news_narrative_rejects_hangul_in_english_output() -> None:
    title = "실적 발표"
    paragraphs = ("매출이 증가했다.",)
    source_hash = _hash(canonical_news_source(title, paragraphs, "FULL_ARTICLE"))
    responses = FakeResponses(
        SimpleNamespace(
            translated_paragraphs=("매출 increased.",),
            what="Revenue increased.",
            why="The source does not state a reason.",
            impact="The source does not state an impact.",
        )
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            _service(responses).translate_news_narrative(
                source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v1"
            )
        )

    assert captured.value.code == "AI_INVALID_OUTPUT"


def test_news_segment_schema_accepts_provider_text_for_service_validation() -> None:
    parsed = _StructuredNewsSegment.model_validate(
        {"translated_text": "The company strengthened 전문 역량."}
    )

    assert parsed.translated_text.endswith("역량.")


def test_long_news_narrative_uses_one_request_per_bounded_segment() -> None:
    title = "장문 기사"
    paragraphs = ("가" * 10_000, "나" * 10_000, "다" * 2_000)
    source_hash = _hash(canonical_news_source(title, paragraphs, "FULL_ARTICLE"))
    responses = SingleNewsResponse()

    result = asyncio.run(
        _service(responses).translate_news_narrative(
            source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v2"
        )
    )

    assert len(result.translated_paragraphs) == len(paragraphs)
    assert responses.calls == 6
    assert {arguments["max_output_tokens"] for arguments in responses.history} == {
        128_000,
    }


def test_many_short_news_paragraphs_keep_one_translation_per_paragraph() -> None:
    title = "다문단 기사"
    paragraphs = tuple(f"문단 {index} " + "가" * 80 for index in range(41))
    source_hash = _hash(canonical_news_source(title, paragraphs, "FULL_ARTICLE"))
    responses = SingleNewsResponse()

    result = asyncio.run(
        _service(responses).translate_news_narrative(
            source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v2"
        )
    )

    assert len(result.translated_paragraphs) == len(paragraphs)
    assert responses.calls == 42


def test_disclosure_section_rejects_missing_table_items() -> None:
    table = json.dumps({"rows": [["매출", 100]]}, ensure_ascii=False)
    source_hash = _hash(canonical_disclosure_section("재무 정보", "매출 현황", table))
    responses = FakeResponses(
        SimpleNamespace(
            translated_heading="Financial Information",
            translated_text="Revenue status",
            translated_table_items=(),
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
    responses = FakeResponses(
        SimpleNamespace(
            translated_heading="Financial Information",
            translated_text="Revenue status",
            translated_table_items=(SimpleNamespace(id="value-0", translated_text="Revenue"),),
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

    assert json.loads(result.translated_table_data_json or "null") == {
        "rows": [["Revenue", 100, True, None]]
    }
    assert responses.arguments["timeout"] == 90.0
    assert responses.arguments["max_output_tokens"] == 128_000


def test_disclosure_section_rejects_hangul_in_english_output() -> None:
    source_hash = _hash(canonical_disclosure_section("제목", "본문", None))
    responses = FakeResponses(
        SimpleNamespace(
            translated_heading="한국항공우주 / Contract",
            translated_text="English body",
            translated_table_items=None,
        )
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            _service(responses).translate_disclosure_section(
                source_hash, "제목", "본문", None, "en", "section-v1"
            )
        )

    assert captured.value.code == "AI_INVALID_OUTPUT"


class FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.arguments: dict[str, object] = {}
        self.calls = 0

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.calls += 1
        self.arguments = arguments
        payload = json.loads(str(arguments["input"]))
        if "source_text" in payload and hasattr(self.parsed, "translated_paragraphs"):
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    translated_text=self.parsed.translated_paragraphs[payload["segment_index"]]
                )
            )
        if "translated_paragraphs" in payload and hasattr(self.parsed, "what"):
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    what=self.parsed.what,
                    why=self.parsed.why,
                    impact=self.parsed.impact,
                )
            )
        return SimpleNamespace(output_parsed=self.parsed)


class FailingResponses:
    def __init__(self, exception: OpenAIError) -> None:
        self.exception = exception

    async def parse(self, **arguments: object) -> SimpleNamespace:
        del arguments
        raise self.exception


class SingleNewsResponse:
    def __init__(self) -> None:
        self.calls = 0
        self.arguments: dict[str, object] = {}
        self.history: list[dict[str, object]] = []

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.calls += 1
        self.arguments = arguments
        self.history.append(arguments)
        payload = json.loads(str(arguments["input"]))
        if "source_text" in payload:
            parsed = SimpleNamespace(
                translated_text=f"Translated segment {payload['segment_index'] + 1}."
            )
        else:
            parsed = SimpleNamespace(
                what="The company announced an update.",
                why="The source states the reason.",
                impact="The source describes a potential impact.",
            )
        return SimpleNamespace(output_parsed=parsed)


def _service(
    responses: FakeResponses | FailingResponses | SingleNewsResponse,
) -> TranslationService:
    return TranslationService(
        SimpleNamespace(responses=responses),
        Settings(environment="test", translation_model="translation-test-model"),
    )


def _title(identifier: str, source_text: str) -> TitleSource:
    return TitleSource(identifier, _hash(source_text), source_text)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
