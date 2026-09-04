import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from k_market_ai.core.errors import AppError
from k_market_ai.translations.news_stream import NewsBundle, completed_summary, stream_news_bundle
from k_market_ai.translations.service import _restore_currency_amounts, canonical_news_source

SUMMARY = {
    "en": {
        "what": "The company announced an investment.",
        "why": "The source cites expansion.",
        "impact": "The source does not state an impact.",
    },
    "ko": {
        "what": "회사가 투자를 발표했다.",
        "why": "원문은 증설을 이유로 제시했다.",
        "impact": "원문에 영향이 명시되지 않았다.",
    },
}


class FakeStream:
    def __init__(self, bundle):
        self.bundle = bundle

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def __aiter__(self):
        yield SimpleNamespace(type="response.output_text.delta", delta='{"summaries":')
        yield SimpleNamespace(
            type="response.output_text.delta",
            delta=json.dumps(self.bundle["summaries"], ensure_ascii=False),
        )
        yield SimpleNamespace(
            type="response.output_text.delta",
            delta=',"items":' + json.dumps(self.bundle["items"]) + "}",
        )

    async def get_final_response(self):
        return SimpleNamespace(
            status="completed", output_parsed=NewsBundle.model_validate(self.bundle)
        )


def generate(items):
    calls = []
    bundle = {"summaries": SUMMARY, "items": items}

    def stream(**kwargs):
        calls.append(kwargs)
        return FakeStream(bundle)

    source = ["회사가 투자를 발표했다.", "증설이 목적이다."]
    canonical = canonical_news_source("투자 발표", source, "FULL_ARTICLE")
    events = stream_news_bundle(
        SimpleNamespace(responses=SimpleNamespace(stream=stream)),
        model="gpt-5-nano",
        source_hash=hashlib.sha256(canonical.encode()).hexdigest(),
        title="투자 발표",
        paragraphs=source,
        content_availability="FULL_ARTICLE",
        translation_version="news-bilingual-v1",
        request_timeout=180,
    )
    return calls, events


def test_one_request_streams_both_summaries_before_complete_body():
    calls, events = generate(
        [
            {"id": "segment-0", "translated_text": "The company announced an investment."},
            {"id": "segment-1", "translated_text": "It cited expansion."},
        ]
    )

    async def run():
        early = await anext(events)
        assert early["type"] == "progress"
        assert early["result"]["summaryReady"] is True
        assert early["result"]["bodyReady"] is False
        assert set(early["result"]["summaries"]) == {"en", "ko"}
        final = await anext(events)
        assert final["type"] == "complete"
        assert final["result"]["bodyReady"] is True
        assert len(final["result"]["translatedParagraphs"]) == 2
        with pytest.raises(StopAsyncIteration):
            await anext(events)

    asyncio.run(run())
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-5-nano"
    assert calls[0]["store"] is False
    assert calls[0]["text"]["format"]["strict"] is True
    assert "text_format" not in calls[0]
    assert "max_output_tokens" not in calls[0]


@pytest.mark.parametrize(
    "items",
    [
        [
            {"id": "segment-0", "translated_text": "영문이 아니다."},
            {"id": "segment-1", "translated_text": "Expansion."},
        ],
        [
            {"id": "segment-0", "translated_text": "Investment."},
            {"id": "segment-0", "translated_text": "Expansion."},
        ],
    ],
)
def test_invalid_body_never_becomes_ready_but_keeps_verified_summary(items):
    calls, events = generate(items)

    async def run():
        assert (await anext(events))["type"] == "progress"
        with pytest.raises(AppError) as error:
            await anext(events)
        assert error.value.code == "AI_INVALID_OUTPUT"

    asyncio.run(run())
    assert len(calls) == 1


def test_incomplete_summary_is_never_published():
    value = '{"summaries":' + json.dumps(SUMMARY)
    assert completed_summary(value[:-1]) is None
    assert completed_summary(value) is not None


def test_provider_interruption_keeps_summary_without_storing_a_partial_body():
    calls = []

    class InterruptedStream(FakeStream):
        async def __aiter__(self):
            yield SimpleNamespace(
                type="response.output_text.delta",
                delta='{"summaries":' + json.dumps(SUMMARY) + ',"items":[',
            )
            yield SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                ),
            )

    def stream(**kwargs):
        calls.append(kwargs)
        return InterruptedStream(None)

    source = ["투자를 발표했다."]
    canonical = canonical_news_source("투자", source, "FULL_ARTICLE")

    async def run():
        events = stream_news_bundle(
            SimpleNamespace(responses=SimpleNamespace(stream=stream)),
            model="gpt-5-nano",
            source_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            title="투자",
            paragraphs=source,
            content_availability="FULL_ARTICLE",
            translation_version="news-bilingual-v1",
            request_timeout=30,
        )
        assert (await anext(events))["type"] == "progress"
        with pytest.raises(AppError) as error:
            await anext(events)
        assert error.value.code == "AI_GENERATION_INCOMPLETE"

    asyncio.run(run())
    assert len(calls) == 1
    assert "max_output_tokens" not in calls[0]


def test_missing_currency_is_not_appended_as_an_unrelated_fallback():
    with pytest.raises(AppError):
        _restore_currency_amounts("매출 1조원", "Revenue increased.")


def test_equivalent_full_won_amount_is_not_rejected_as_missing_token():
    from k_market_ai.translations.service import _normalize_english_output

    result = _restore_currency_amounts("137만 9000원", "The price was KRW 1,379,000.")
    assert _normalize_english_output(result) == "The price was KRW 1.379 million."
    with pytest.raises(AppError):
        _restore_currency_amounts("137만 9000원", "The price was KRW 1,378,000.")
    with pytest.raises(AppError):
        _restore_currency_amounts("137만 9000원", "The price was USD 1,379,000.")
    assert _normalize_english_output("KRW 25.50") == "KRW 25.5"


def test_body_retry_reuses_verified_bilingual_summary_without_regeneration():
    calls = []
    body = {
        "items": [{"id": "segment-0", "translated_text": "The company announced an investment."}]
    }

    class BodyStream(FakeStream):
        async def __aiter__(self):
            yield SimpleNamespace(type="response.output_text.delta", delta=json.dumps(body))

        async def get_final_response(self):
            return SimpleNamespace(status="completed")

    def stream(**kwargs):
        calls.append(kwargs)
        return BodyStream(body)

    source = ["회사가 투자를 발표했다."]
    canonical = canonical_news_source("투자 발표", source, "FULL_ARTICLE")

    async def run():
        return [
            event
            async for event in stream_news_bundle(
                SimpleNamespace(responses=SimpleNamespace(stream=stream)),
                model="gpt-5-nano",
                source_hash=hashlib.sha256(canonical.encode()).hexdigest(),
                title="투자 발표",
                paragraphs=source,
                content_availability="FULL_ARTICLE",
                translation_version="news-bilingual-v1",
                request_timeout=30,
                cached_summaries=SUMMARY,
            )
        ]

    events = asyncio.run(run())
    assert len(events) == 1 and events[0]["type"] == "complete"
    assert events[0]["result"]["summaries"] == SUMMARY
    assert events[0]["result"]["bodyReady"] is True
    assert "summaries" not in calls[0]["text"]["format"]["schema"]["properties"]
    assert calls[0]["text"]["format"]["schema"]["properties"]["items"]["minItems"] == 1


@pytest.mark.parametrize(
    "translated",
    [
        "At least KRW 10 million; KRW 700, KRW 7,000, or KRW 70,000 will be paid.",
        "At least 10 million won; 700 won, 7,000 won, or 70,000 won will be paid.",
        "At least 10 million KRW; 700 KRW, 7,000 KRW, or 70,000 KRW will be paid.",
    ],
)
def test_currency_lists_preserve_each_amount_without_moving_the_next_currency(translated):
    from k_market_ai.translations.service import _normalize_english_output

    source = "자산을 1000만원 이상 늘리면 700원, 7000원, 7만원 중 지급한다."
    normalized = _normalize_english_output(_restore_currency_amounts(source, translated))
    assert normalized == (
        "At least KRW 10 million; KRW 700, KRW 7,000, or KRW 70,000 will be paid."
    )
    assert _normalize_english_output(normalized) == normalized


@pytest.mark.parametrize(
    "translated",
    [
        "KRW 10 million; KRW 700, KRW 70,000.",
        "KRW 10 million; KRW 700, KRW 7,001, KRW 70,000.",
        "KRW 10 million; KRW 700, USD 7,000, KRW 70,000.",
        "KRW 10 million; KRW 700, KRW 7,000,000, KRW 70,000.",
    ],
)
def test_currency_list_fix_does_not_accept_missing_changed_or_different_currency(translated):
    with pytest.raises(AppError):
        _restore_currency_amounts("1000만원, 700원, 7000원, 7만원", translated)


def test_decimal_currency_lists_keep_punctuation_and_exact_values():
    from k_market_ai.translations.service import _normalize_english_output

    translated = "KRW 2.6135 trillion, KRW 3.2 billion. KRW 25.50, KRW 700."
    assert _normalize_english_output(translated) == (
        "KRW 2.6135 trillion, KRW 3.2 billion. KRW 25.5, KRW 700."
    )


def test_valid_direct_currency_output_is_not_duplicated():
    assert (
        _restore_currency_amounts("매출 1조원", "Revenue reached KRW 1 trillion.")
        == "Revenue reached KRW 1 trillion."
    )


@pytest.mark.parametrize("value", ["5,000 KRW", "5000 won", "KRW 5000"])
def test_equivalent_currency_position_and_separator_are_not_rejected(value):
    from k_market_ai.translations.service import _normalize_english_output

    assert (
        _normalize_english_output(_restore_currency_amounts("5000원 상승", f"Up {value}."))
        == "Up KRW 5,000."
    )


def test_shared_won_unit_in_one_sentence_preserves_all_amounts():
    translated = "KRW 70 trillion in dividends (30 trillion and 40 trillion respectively)."
    assert (
        _restore_currency_amounts("3분기 30조원, 4분기 40조원, 총 70조원", translated) == translated
    )


@pytest.mark.parametrize(
    "translated",
    [
        "KRW 70 trillion plus 30 trillion dollars.",
        "KRW 70 trillion plus 30 trillion shares.",
        "KRW 70 trillion.",
    ],
)
def test_shared_unit_does_not_accept_other_currencies_quantities_or_missing_amounts(translated):
    with pytest.raises(AppError):
        _restore_currency_amounts("30조원 및 70조원", translated)


def test_sentence_units_return_to_original_paragraphs():
    source = ["회사가 투자를 발표했다. 공장을 건설한다.", "내년 가동한다."]
    bundle = {
        "summaries": SUMMARY,
        "items": [
            {"id": "segment-0", "translated_text": "The company announced investment."},
            {"id": "segment-1", "translated_text": "It will build a factory."},
            {"id": "segment-2", "translated_text": "Operations begin next year."},
        ],
    }

    async def run():
        canonical = canonical_news_source("투자", source, "FULL_ARTICLE")
        client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kw: FakeStream(bundle)))
        return [
            event
            async for event in stream_news_bundle(
                client,
                model="gpt-5-nano",
                source_hash=hashlib.sha256(canonical.encode()).hexdigest(),
                title="투자",
                paragraphs=source,
                content_availability="FULL_ARTICLE",
                translation_version="news-bilingual-v1",
                request_timeout=30,
            )
        ]

    result = asyncio.run(run())[-1]["result"]
    assert result["translatedParagraphs"] == [
        "The company announced investment. It will build a factory.",
        "Operations begin next year.",
    ]


@pytest.mark.parametrize(
    "source, expected",
    [
        ("34조 2천억 원", "KRW 34.2 trillion"),
        ("16조 8천억원", "KRW 16.8 trillion"),
        ("2천500억원", "KRW 250 billion"),
        ("4천만원", "KRW 40 million"),
    ],
)
def test_korean_small_units_are_part_of_the_amount(source, expected):
    from k_market_ai.translations.service import _protect_currency_amounts

    protected, amounts = _protect_currency_amounts(source)
    assert protected == "__KRW_AMOUNT_0__"
    assert amounts == (("__KRW_AMOUNT_0__", expected),)


def test_legal_articles_are_not_krw_amounts():
    from k_market_ai.translations.service import _protect_currency_amounts

    source = "제16조 제1항과 12조에 따라 제2-2조 제2항을 적용한다."
    assert _protect_currency_amounts(source) == (source, ())


@pytest.mark.parametrize("unit", ["t", "톤", "GWh", "㎡", "평", "주", "가구", "세대", "곳"])
def test_non_currency_units_never_receive_krw(unit):
    from k_market_ai.translations.service import (
        _canonicalize_non_krw_quantities,
        _protect_currency_amounts,
    )

    source = f"생산능력 32만{unit}"
    assert _protect_currency_amounts(source) == (source, ())
    assert _canonicalize_non_krw_quantities(source) == f"생산능력 320,000{unit}"


def test_currency_token_one_does_not_consume_token_ten():
    from k_market_ai.translations.service import _protect_currency_amounts

    source = " ".join(f"{i + 1}억원" for i in range(12))
    protected, _ = _protect_currency_amounts(source)
    restored = _restore_currency_amounts(source, protected)
    assert "AMOUNT" not in restored
    assert "KRW 1.1 billion" in restored


def test_ranked_names_and_household_counts_are_not_monetary_requirements():
    from k_market_ai.translations.service import _protect_currency_amounts

    source = "1 조국혁신당 조국, 3 조건. 소득 하위 20% 중 310만 가구."
    assert _protect_currency_amounts(source) == (source, ())
    translated = "1 Cho Kuk, 3 conditions. Among the bottom 20%, 3.1 million households."
    assert _restore_currency_amounts(source, translated) == translated
    assert _protect_currency_amounts("투자 1조와 매출 2조는 증가했다.")[1] == (
        ("__KRW_AMOUNT_0__", "KRW 1 trillion"),
        ("__KRW_AMOUNT_1__", "KRW 2 trillion"),
    )


def test_stream_preserves_korean_amount_and_does_not_cut_english_currency_unit():
    source = ["회사가 34조 2천억 원을 투자한다."]
    summaries = {
        **SUMMARY,
        "en": {**SUMMARY["en"], "what": "Investment totals __KRW_SEGMENT_0_AMOUNT_0__."},
        "ko": {**SUMMARY["ko"], "what": "회사가 __KRW_SEGMENT_0_AMOUNT_0__을 투자한다."},
    }
    bundle = {
        "summaries": summaries,
        "items": [
            {"id": "segment-0", "translated_text": "Investment totals __KRW_SEGMENT_0_AMOUNT_0__."}
        ],
    }
    calls = []

    def stream(**kwargs):
        calls.append(kwargs)
        return FakeStream(bundle)

    async def run():
        canonical = canonical_news_source("투자", source, "FULL_ARTICLE")
        events = stream_news_bundle(
            SimpleNamespace(responses=SimpleNamespace(stream=stream)),
            model="gpt-5-nano",
            source_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            title="투자",
            paragraphs=source,
            content_availability="FULL_ARTICLE",
            translation_version="news-bilingual-v1",
            request_timeout=180,
        )
        results = [event async for event in events]
        assert results[0]["result"]["summaries"]["ko"]["what"] == "회사가 34조 2천억 원을 투자한다."
        assert results[1]["result"]["translatedParagraphs"] == [
            "Investment totals KRW 34.2 trillion."
        ]

    asyncio.run(run())
    assert len(calls) == 1
