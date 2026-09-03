import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Literal

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

from k_market_ai.translations.service import _contains_invalid_english

AnswerLanguage = Literal["en", "ko"]
AnswerLocale = Literal["en", "ko", "auto"]
KOREAN_SCRIPT_SCHEMA_PATTERN = r"[\s\S]*[가-힣][\s\S]*"


@lru_cache(maxsize=1)
def _detector() -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.KOREAN).build()


def resolve_answer_language(question: str, policy: AnswerLocale = "auto") -> AnswerLanguage:
    if policy != "auto":
        return policy
    # 인용문·URL을 제외한 질문만 판정하고 사용자 문장은 캐시하지 않는다.
    text = re.sub(r"```[\s\S]*?```|`[^`]*`|https?://\S+", " ", question)
    text = re.sub(r'"[^"\n]*"|“[^”]*”|「[^」]*」|(?<!\w)\x27[^\x27\n]*\x27(?!\w)', " ", text)
    text = re.sub(r"(?m)^\s*>.*$", " ", text).strip()
    if not re.search(r"[a-zA-Z가-힣]", text):
        text = question
    prefix, separator, _ = text.partition(":")
    if separator and re.search(
        r"\b(explain|summarize|translate)\b|설명|요약|해석|번역", prefix, re.I
    ):
        text = prefix
    if re.search(r"[가-힣]", text):
        # 영문 고유명사와 한국어 질문 종결 표현이 섞여도 회사명이 문장 언어를 덮지 않는다.
        text = re.sub(r"\b[A-Z][A-Za-z0-9.&-]*(?:\s+[A-Z][A-Za-z0-9.&-]*)+\b", " ", text)
        if re.search(r"(?:[가-힣]+(?:요|까|줘|죠|은|는|란|인지|인가)|어때)\s*[?!.…]*$", text):
            return "ko"
    return "ko" if _detector().detect_language_of(text) == Language.KOREAN else "en"


def answer_language_instructions(locale: AnswerLanguage) -> str:
    language = {"en": "English", "ko": "Korean"}[locale]
    return (
        f"\nOutput language: {language} ({locale}). Write every user-facing answer, refusal, "
        "disclaimer and suggested room name in this language. This language was determined "
        "from the current question alone. Selected text, evidence and prior conversation "
        "do not change the output language. Preserve facts, amounts and citation IDs. "
        + (
            "Translate Korean labels and company suffixes, including (주) and ㈜, into English. "
            "Use lock-up period for 의무보유 기간, lock-up release date for 의무보유 해제일, "
            "SK Innovation for SK이노베이션, and SK Inc. for SK(주). "
            "Do not copy Korean characters, even in quotations or parentheses."
            if locale == "en"
            else "Use natural Korean sentences and original Korean labels. Company names, "
            "tickers, dates and citation IDs may keep their original spelling."
        )
    )


def valid_answer_language(values: Iterable[str | None], locale: AnswerLanguage) -> bool:
    return all(
        not value
        or (
            not _contains_invalid_english(value)
            if locale == "en"
            else re.search(r"[가-힣]", value) is not None
        )
        for value in values
    )
