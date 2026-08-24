from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TitleSource:
    id: str
    source_hash: str
    source_text: str


@dataclass(frozen=True, slots=True)
class TitleTranslation:
    id: str
    source_hash: str
    translated_text: str


@dataclass(frozen=True, slots=True)
class TitleTranslationBatch:
    items: tuple[TitleTranslation, ...]
    target_locale: str
    translation_version: str
    model: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class NewsNarrative:
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


@dataclass(frozen=True, slots=True)
class DisclosureSectionTranslation:
    source_hash: str
    translated_heading: str | None
    translated_text: str | None
    translated_table_data_json: str | None
    target_locale: str
    translation_version: str
    model: str
    prompt_version: str
