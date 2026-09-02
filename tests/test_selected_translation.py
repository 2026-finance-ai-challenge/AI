import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from k_market_ai.rag.domain.models import SectionKind, SourceSection
from k_market_ai.rag.domain.selected_translation import translated_selection_exists
from k_market_ai.rag.infrastructure.postgres_repository import PostgresRagRepository


def source(text: str = "해제일 2026.09.05", heading: str | None = None) -> SourceSection:
    return SourceSection(uuid4(), uuid4(), 1, 1, SectionKind.TABLE, heading, text)


def canonical(text: str = "해제일 2026.09.05", heading: str | None = None) -> tuple[str, str]:
    value = json.dumps({"heading": heading, "text": text, "table_data_json": None})
    return value, hashlib.sha256(value.encode()).hexdigest()


def test_validates_english_table_selection_against_current_korean_source() -> None:
    value, digest = canonical()
    assert translated_selection_exists(
        source(),
        "Release date 2026.09.05",
        digest,
        value,
        {"translatedTableData": [["Release date", "2026.09.05"]]},
    )


def test_validates_ready_paragraph_selection() -> None:
    value, digest = canonical()
    assert translated_selection_exists(
        source(),
        "September 5",
        digest,
        value,
        {"translatedText": "Release date: September\n5"},
    )


@pytest.mark.parametrize("selected", ["", "Buy now", "Release date 2026.09.06"])
def test_rejects_missing_or_unrelated_selection(selected: str) -> None:
    value, digest = canonical()
    assert not translated_selection_exists(
        source(),
        selected,
        digest,
        value,
        {"translatedTableData": [["Release date", "2026.09.05"]]},
    )


def test_rejects_cache_for_other_source_heading_or_hash() -> None:
    value, digest = canonical()
    result = {"translatedText": "Release date"}
    assert not translated_selection_exists(source("다른 공시"), "Release", digest, value, result)
    assert not translated_selection_exists(
        source(heading="다른 제목"), "Release", digest, value, result
    )
    assert not translated_selection_exists(source(), "Release", "a" * 64, value, result)


@pytest.mark.parametrize(
    "result", [None, {}, {"translatedTableData": ["Release"]}, {"translatedTableData": [[1]]}]
)
def test_rejects_malformed_result(result: object) -> None:
    value, digest = canonical()
    assert not translated_selection_exists(source(), "Release", digest, value, result)


def test_repository_scopes_ready_cache_by_hash_and_current_section() -> None:
    section = source()
    value, digest = canonical()
    pool = MagicMock()
    connection = pool.connection.return_value.__aenter__.return_value
    cursor = AsyncMock()
    connection.execute = AsyncMock(return_value=cursor)
    cursor.fetchone.return_value = (value, {"translatedText": "Release date"})
    repository = PostgresRagRepository(pool)
    repository.load_current_sections = AsyncMock(return_value=[section])
    assert asyncio.run(
        repository.selected_text_exists("20260902800513", section.id, "Release", digest)
    )
    sql, params = connection.execute.call_args.args
    assert "status = 'READY'" in sql
    assert "translation_version = 'disclosure-section-v4'" in sql
    assert params == (digest,)
    repository.load_current_sections.assert_awaited_once_with("20260902800513")
    connection.execute.reset_mock()
    assert not asyncio.run(
        repository.selected_text_exists("20260902800513", uuid4(), "Release", digest)
    )
    connection.execute.assert_not_called()


def test_repository_rejects_unavailable_translation_without_model_call() -> None:
    section = source()
    pool = MagicMock()
    connection = pool.connection.return_value.__aenter__.return_value
    cursor = AsyncMock()
    connection.execute = AsyncMock(return_value=cursor)
    cursor.fetchone.return_value = None
    repository = PostgresRagRepository(pool)
    repository.load_current_sections = AsyncMock(return_value=[section])
    assert not asyncio.run(
        repository.selected_text_exists("20260902800513", section.id, "Release", "a" * 64)
    )
