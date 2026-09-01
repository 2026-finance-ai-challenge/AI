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
    _canonicalize_non_krw_quantities,
    _currency_conversions,
    _restore_currency_amounts,
    _StructuredNewsSegmentItem,
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
    assert responses.arguments["max_output_tokens"] == 16_384


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


def test_korean_currency_conversion_preserves_round_and_compound_units() -> None:
    assert _currency_conversions("170조원, 169조6022억원, 4000억원, 25만4000원") == [
        {"source_text": "170조원", "english_text": "KRW 170 trillion"},
        {"source_text": "169조6022억원", "english_text": "KRW 169.6022 trillion"},
        {"source_text": "4000억원", "english_text": "KRW 400 billion"},
        {"source_text": "25만4000원", "english_text": "KRW 254,000"},
    ]


def test_korean_currency_conversion_ignores_share_counts_and_foreign_currency() -> None:
    source = "3301만 6411주, 10만 8590주, 14억 6296만스위스프랑, 투자 1조1000억"

    assert _currency_conversions(source) == [
        {"source_text": "1조1000억", "english_text": "KRW 1.1 trillion"}
    ]


def test_korean_magnitude_quantities_are_canonicalized_before_translation() -> None:
    source = "3301만 6411주, 10만 8590주, 14억 6296만 7000스위스프랑, 투자 1조1000억"

    assert _canonicalize_non_krw_quantities(source) == (
        "33,016,411주, 108,590주, 1,462,967,000스위스프랑, 투자 1조1000억"
    )


def test_currency_restoration_accepts_provider_token_punctuation_variants() -> None:
    source = "누적 1조원, 당일 4867억원"
    translated = "Cumulative KRW_AMOUNT_0 and daily __KRW_AMOUNT_1__sales"

    assert _restore_currency_amounts(source, translated) == (
        "Cumulative KRW 1 trillion and daily KRW 486.7 billionsales"
    )


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


def test_english_title_batch_preserves_samjeonnix_and_currency_spacing() -> None:
    source = _title("T1", "'삼전닉스' 성과급에 지갑 열렸다…지역 소비 1조1000억 증가")
    responses = FakeResponses(
        SimpleNamespace(
            items=(
                SimpleNamespace(
                    id="T1",
                    source_hash=source.source_hash,
                    translated_text=(
                        "'__TERM_SAMJEONNIX__' incentives lift spending; "
                        "consumption __KRW_AMOUNT_0__rises"
                    ),
                ),
            )
        )
    )

    result = asyncio.run(_service(responses).translate_titles((source,), "en", "title-v1"))

    assert result.items[0].translated_text == (
        "'Samjeonnix' incentives lift spending; consumption KRW 1.1 trillion rises"
    )
    payload = json.loads(str(responses.arguments["input"]))
    assert payload["items"][0]["source_text"].startswith("'__TERM_SAMJEONNIX__'")
    assert payload["items"][0]["protected_term_tokens"] == ["__TERM_SAMJEONNIX__"]


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
    assert responses.arguments["timeout"] == 180.0


def test_korean_news_narrative_uses_original_body_and_generates_korean_summary() -> None:
    title = "실적 발표"
    paragraphs = ("매출이 증가했다.", "회사는 해외 수요를 원인으로 설명했다.")
    source_hash = _hash(canonical_news_source(title, paragraphs, "FULL_ARTICLE"))
    responses = FakeResponses(
        SimpleNamespace(
            what="회사 매출이 증가했다.",
            why="해외 수요가 증가했다.",
            impact="향후 실적 개선이 예상된다.",
        )
    )

    result = asyncio.run(
        _service(responses).translate_news_narrative(
            source_hash,
            title,
            paragraphs,
            "FULL_ARTICLE",
            "ko",
            "news-v10",
        )
    )

    assert result.translated_paragraphs == paragraphs
    assert result.target_locale == "ko"
    assert result.what == "회사 매출이 증가했다."
    assert responses.calls == 1


def test_news_narrative_rejects_field_label_placeholders_without_fallback() -> None:
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

    with pytest.raises(AppError) as captured:
        asyncio.run(
            _service(responses).translate_news_narrative(
                source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v3"
            )
        )

    assert captured.value.code == "AI_INVALID_OUTPUT"
    assert responses.calls == 2


def test_news_narrative_rejects_non_english_summary_without_fallback() -> None:
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

    with pytest.raises(AppError) as captured:
        asyncio.run(
            _service(responses).translate_news_narrative(
                source_hash, title, paragraphs, "FULL_ARTICLE", "en", "news-v6"
            )
        )

    assert captured.value.code == "AI_INVALID_OUTPUT"
    assert responses.calls == 2


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


def test_news_segment_item_schema_accepts_provider_text_for_service_validation() -> None:
    parsed = _StructuredNewsSegmentItem.model_validate(
        {"id": "segment-0", "translated_text": "The company strengthened 전문 역량."}
    )

    assert parsed.translated_text.endswith("역량.")


def test_long_news_narrative_batches_bounded_segments() -> None:
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
    assert responses.calls == 2
    assert {arguments["max_output_tokens"] for arguments in responses.history} == {
        128_000,
    }


def test_many_short_news_paragraphs_use_bounded_batches() -> None:
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
    assert responses.calls == 3


def test_disclosure_section_rejects_missing_table_items_without_fallback() -> None:
    table = json.dumps({"rows": [["매출", 100]]}, ensure_ascii=False)
    source_hash = _hash(canonical_disclosure_section("재무 정보", "매출 현황", table))
    responses = DisclosureResponses(
        {"재무 정보": "Financial Information"},
        (),
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
    responses = DisclosureResponses(
        {"재무 정보": "Financial Information"},
        (SimpleNamespace(id="value-0", translated_text="Revenue"),),
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
    assert result.translated_text == "Revenue"
    assert responses.arguments["timeout"] == 90.0
    assert responses.arguments["max_output_tokens"] == 16_384


def test_disclosure_section_rejects_hangul_in_english_output() -> None:
    source_hash = _hash(canonical_disclosure_section("제목", "본문", None))
    responses = DisclosureResponses(
        {"제목": "한국항공우주 / Contract", "본문": "English body"},
        (),
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            _service(responses).translate_disclosure_section(
                source_hash, "제목", "본문", None, "en", "section-v1"
            )
        )

    assert captured.value.code == "AI_INVALID_OUTPUT"


def test_disclosure_table_translation_is_bounded_and_preserves_ascii_cells() -> None:
    rows = [[f"항목 {index}", str(index)] for index in range(20)]
    table = json.dumps(rows, ensure_ascii=False)
    source_hash = _hash(canonical_disclosure_section(None, "표 본문", table))
    responses = EchoDisclosureResponses()

    result = asyncio.run(
        _service(responses).translate_disclosure_section(
            source_hash, None, "표 본문", table, "en", "section-v2"
        )
    )

    translated = json.loads(result.translated_table_data_json or "null")
    assert translated[0] == ["Item 0", "0"]
    assert translated[-1] == ["Item 19", "19"]
    assert responses.batch_sizes == [18, 2]


class FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.arguments: dict[str, object] = {}
        self.calls = 0

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.calls += 1
        self.arguments = arguments
        payload = json.loads(str(arguments["input"]))
        if "items" in payload and hasattr(self.parsed, "translated_paragraphs"):
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    items=tuple(
                        SimpleNamespace(
                            id=item["id"],
                            translated_text=self.parsed.translated_paragraphs[
                                int(item["id"].removeprefix("segment-"))
                            ],
                        )
                        for item in payload["items"]
                    )
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
        if "items" in payload:
            parsed = SimpleNamespace(
                items=tuple(
                    SimpleNamespace(
                        id=item["id"],
                        translated_text=(
                            f"Translated segment {int(item['id'].removeprefix('segment-')) + 1}."
                        ),
                    )
                    for item in payload["items"]
                )
            )
        else:
            parsed = SimpleNamespace(
                what="The company announced an update.",
                why="The source states the reason.",
                impact="The source describes a potential impact.",
            )
        return SimpleNamespace(output_parsed=parsed)


class DisclosureResponses:
    def __init__(
        self,
        text_outputs: dict[str, str],
        table_items: tuple[SimpleNamespace, ...],
    ) -> None:
        self.text_outputs = text_outputs
        self.table_items = table_items
        self.arguments: dict[str, object] = {}
        self.calls = 0

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.calls += 1
        self.arguments = arguments
        payload = json.loads(str(arguments["input"]))
        if "items" in payload:
            return SimpleNamespace(output_parsed=SimpleNamespace(items=self.table_items))
        return SimpleNamespace(
            output_parsed=SimpleNamespace(
                translated_text=self.text_outputs[str(payload["source_text"])]
            )
        )


class EchoDisclosureResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}
        self.batch_sizes: list[int] = []

    async def parse(self, **arguments: object) -> SimpleNamespace:
        self.arguments = arguments
        payload = json.loads(str(arguments["input"]))
        items = payload["items"]
        self.batch_sizes.append(len(items))
        return SimpleNamespace(
            output_parsed=SimpleNamespace(
                items=tuple(
                    SimpleNamespace(
                        id=item["id"],
                        translated_text=item["source_text"].replace("항목", "Item"),
                    )
                    for item in items
                )
            )
        )


def _service(
    responses: (
        FakeResponses
        | FailingResponses
        | SingleNewsResponse
        | DisclosureResponses
        | EchoDisclosureResponses
    ),
) -> TranslationService:
    return TranslationService(
        SimpleNamespace(responses=responses),
        Settings(environment="test", translation_model="translation-test-model"),
    )


def _title(identifier: str, source_text: str) -> TitleSource:
    return TitleSource(identifier, _hash(source_text), source_text)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
