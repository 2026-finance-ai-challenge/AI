import asyncio
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from k_market_ai.core.config import Settings
from k_market_ai.main import create_app
from k_market_ai.rag.application.ask_disclosure import (
    AskDisclosureHandler,
    _financial_columns,
    _financial_period_dates,
)
from k_market_ai.rag.domain.models import FilingCandidate, SearchHit, SectionKind, SourceSection


def repository(indexed: bool):
    section_id, document_id = uuid4(), uuid4()
    filing = FilingCandidate(
        "20260814003699",
        "005930",
        "반기보고서 (2026.06)",
        date(2026, 8, 14),
        datetime(2026, 8, 14, tzinfo=UTC),
    )
    hit = SearchHit(
        uuid4(),
        document_id,
        2,
        (section_id,),
        1,
        1,
        "연결 손익계산서",
        "단위 백만원 매출액 100 영업이익 20",
        0.8,
        2,
    )
    return SimpleNamespace(
        evidence_candidates=AsyncMock(return_value=[filing]),
        search=AsyncMock(return_value=[hit] if indexed else []),
        load_current_sections=AsyncMock(
            return_value=[
                SourceSection(
                    section_id,
                    document_id,
                    2,
                    1,
                    SectionKind.TABLE,
                    hit.heading,
                    hit.content,
                )
            ]
        ),
    )


def test_general_retrieval_uses_current_vectors_without_second_answer_call():
    repo = repository(True)
    answer = SimpleNamespace(answer=AsyncMock())
    embedding = SimpleNamespace(model="test", embed=AsyncMock(return_value=[(1.0,)]))
    result = asyncio.run(
        AskDisclosureHandler(repo, embedding, answer).retrieve(
            ["005930"],
            "recent Samsung filings",
            None,
            None,
            False,
        )
    )
    assert len(result) == 1
    assert result[0].retrieval_method == "CURRENT_VECTOR_CHUNKS"
    assert "영업이익 20" in result[0].content
    repo.evidence_candidates.assert_awaited_once_with(["005930"], None, None, False)
    repo.load_current_sections.assert_not_awaited()
    answer.answer.assert_not_awaited()


def test_pending_embeddings_search_current_source_instead_of_old_chunks():
    repo = repository(False)
    answer = SimpleNamespace(answer=AsyncMock())
    embedding = SimpleNamespace(model="test", embed=AsyncMock(return_value=[(1.0,)]))
    result = asyncio.run(
        AskDisclosureHandler(repo, embedding, answer).retrieve(
            ["005930"],
            "최근 매출",
            date(2026, 8, 1),
            date(2026, 8, 31),
            True,
        )
    )
    assert result[0].retrieval_method == "CURRENT_HYBRID_FINANCIAL"
    assert "단위 백만원" in result[0].content
    repo.load_current_sections.assert_awaited_once_with("20260814003699")
    answer.answer.assert_not_awaited()


def test_evidence_endpoint_requires_internal_auth_and_bounds_company_scope():
    token = str(uuid4())
    handler = SimpleNamespace(retrieve=AsyncMock(return_value=[]))
    app = create_app(
        Settings(environment="test", allowed_hosts=["testserver"], service_token=token),
        rag_handler=handler,
    )
    payload = {"stock_codes": ["005930"], "question": "recent earnings", "financials": True}
    headers = {"authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        assert client.post("/internal/v1/disclosures/evidence", json=payload).status_code == 401
        assert (
            client.post(
                "/internal/v1/disclosures/evidence",
                headers=headers,
                json={**payload, "stock_codes": ["005930", "000660", "005380"]},
            ).status_code
            == 422
        )
        result = client.post("/internal/v1/disclosures/evidence", headers=headers, json=payload)
        assert result.status_code == 200 and result.json() == []
    handler.retrieve.assert_awaited_once_with(["005930"], "recent earnings", None, None, True)


def test_financial_evidence_preserves_header_and_period_cell_order():
    repo = repository(True)
    source = repo.load_current_sections.return_value[0]
    repo.load_current_sections.return_value = [
        replace(source, kind=SectionKind.TITLE, text="연결 손익계산서"),
        replace(
            source,
            ordinal=2,
            kind=SectionKind.TEXT,
            text="2026.01.01~06.30 / 2025.01.01~06.30 (단위: 백만원)",
        ),
        replace(
            source,
            ordinal=3,
            text="매출액 100 200 90 180 영업이익 20 40 18 36 순이익 15 30 12 24",
            table_rows=(
                ("", "2026 반기", "2025 반기"),
                ("3개월", "누적", "3개월", "누적"),
                ("매출액", "100", "200", "90", "180"),
            ),
        ),
    ]
    embedding = SimpleNamespace(model="test", embed=AsyncMock(return_value=[(1.0,)]))
    result = asyncio.run(
        AskDisclosureHandler(repo, embedding, SimpleNamespace()).retrieve(
            ["005930"], "recent earnings", None, None, True
        )
    )
    assert result[0].retrieval_method == "CURRENT_STRUCTURED_FINANCIAL"
    assert "2026.01.01~06.30" in result[0].content
    assert '"period":"2026 반기","interval":"누적","rows":[["매출액","200"]]' in result[0].content
    assert '"period":"2025 반기"' not in result[0].content


def test_financial_columns_keep_current_prior_and_duplicate_metric_rows_separate():
    rows = (
        ("　", "제 32 기 반기", "제 31 기 반기"),
        ("3개월", "누적", "3개월", "누적"),
        ("당기순이익(손실)", "17,980", "244,816", "171,815", "372,149"),
        ("　지배기업 소유주지분", "16,332", "188,028", "161,247", "333,098"),
        ("당기총포괄손익", "175,910", "533,367", "339,983", "657,005"),
        ("　지배기업 소유주지분", "169,494", "477,302", "327,473", "606,926"),
    )
    serialized = _financial_columns(rows)
    assert serialized is not None
    columns = [json.loads(line) for line in serialized.splitlines()]
    assert [(c["period"], c["interval"]) for c in columns] == [
        ("제 32 기 반기", "3개월"),
        ("제 32 기 반기", "누적"),
        ("제 31 기 반기", "3개월"),
        ("제 31 기 반기", "누적"),
    ]
    assert columns[1]["rows"][0] == ["당기순이익(손실)", "244,816"]
    assert columns[3]["rows"][0] == ["당기순이익(손실)", "372,149"]
    assert columns[1]["rows"][1][1] == "188,028"
    assert columns[1]["rows"][3][1] == "477,302"
    assert columns[1]["rows"][1][0] == "당기순이익(손실) > 지배기업 소유주지분"
    assert columns[1]["rows"][3][0] == "당기총포괄손익 > 지배기업 소유주지분"
    for i, column in enumerate(columns):
        assert [row[1] for row in column["rows"]] == [row[i + 1] for row in rows[2:]]
    current = _financial_columns(rows, latest_only=True)
    assert current is not None
    assert [json.loads(line)["period"] for line in current.splitlines()] == ["제 32 기 반기"] * 2
    assert "244,816" in current and "372,149" not in current


def test_latest_period_is_chosen_by_explicit_term_not_column_position():
    rows = (
        ("", "제 31 기 반기", "제 32 기 반기"),
        ("3개월", "누적", "3개월", "누적"),
        ("순이익", "1", "2", "3", "4"),
    )
    result = _financial_columns(rows, latest_only=True)
    assert result is not None
    columns = [json.loads(line) for line in result.splitlines()]
    assert [c["rows"][0][1] for c in columns] == ["3", "4"]


def test_mixed_year_and_term_headers_are_not_compared_as_same_scale():
    rows = (
        ("", "제 32 기 반기", "2025 반기"),
        ("3개월", "누적", "3개월", "누적"),
        ("순이익", "1", "2", "3", "4"),
    )
    result = _financial_columns(rows, latest_only=True)
    assert result is not None and len(result.splitlines()) == 4


@pytest.mark.parametrize("interval,start", [("3개월", "2026-04-01"), ("누적", "2026-01-01")])
def test_half_year_and_three_month_periods_have_distinct_start_dates(interval, start):
    context = (
        "제 32 기 반기 2026.01.01 부터 2026.06.30 까지 "
        "제 31 기 반기 2025.01.01 부터 2025.06.30 까지"
    )
    assert _financial_period_dates(context, "제 32 기 반기", interval) == {
        "start_date": start,
        "end_date": "2026-06-30",
    }


@pytest.mark.parametrize(
    "context",
    [
        "제32기 반기 2026.01.01~2026.06.30",
        "제32기 반기 2026.01.01 부터 2026.02.31 까지",
        "제32기 반기 2026.06.30 부터 2026.01.01 까지",
        "제32기 반기 2026.01.01 부터 2026.06.15 까지",
    ],
)
def test_missing_or_ambiguous_quarter_dates_are_not_invented(context):
    assert _financial_period_dates(context, "제32기 반기", "3개월") == {}


@pytest.mark.parametrize(
    "rows",
    [
        (),
        ((), (), ()),
        (("", "당기", "전기"), ("누적", "3개월", "누적", "3개월"), ("매출", "1", "2", "3", "4")),
        (("", "당기", "전기"), ("3개월", "누적", "3개월", "누적"), ("매출", "1", "2")),
        (
            ("항목", "당기", "전기"),
            ("3개월", "누적", "3개월", "누적"),
            ("매출", "1", "2", "3", "4"),
        ),
    ],
)
def test_ambiguous_financial_headers_are_not_guessed(rows):
    assert _financial_columns(rows) is None


def test_duplicate_current_evidence_keeps_latest_filing_but_not_other_company():
    repo = repository(True)
    latest = repo.evidence_candidates.return_value[0]
    repo.evidence_candidates.return_value = [
        latest,
        replace(latest, receipt_number="20260813000001", filed_date=date(2026, 8, 13)),
        replace(latest, receipt_number="20260812000001", stock_code="035720"),
    ]
    embedding = SimpleNamespace(model="test", embed=AsyncMock(return_value=[(1.0,)]))
    result = asyncio.run(
        AskDisclosureHandler(repo, embedding, SimpleNamespace()).retrieve(
            ["005930", "035720"], "compare earnings", None, None, True
        )
    )
    assert [item.filing.receipt_number for item in result] == ["20260814003699", "20260812000001"]


@pytest.mark.parametrize(
    "question,from_date,expected",
    [
        ("Please tell me about recent Kakao's earning", None, 1),
        ("카카오 최근 실적 알려줘", None, 1),
        ("compare recent Kakao earnings", None, 2),
        ("최근 실적 추이", None, 2),
        ("recent Kakao earnings", date(2026, 1, 1), 2),
    ],
)
def test_latest_overview_does_not_mix_older_preliminary_figures(question, from_date, expected):
    repo = repository(True)
    latest = repo.evidence_candidates.return_value[0]
    repo.evidence_candidates.return_value = [
        latest,
        replace(latest, receipt_number="20260813000001"),
    ]
    section = repo.load_current_sections.return_value[0]
    repo.load_current_sections.side_effect = [
        [section],
        [replace(section, text=section.text + " 별도 잠정")],
    ]
    embedding = SimpleNamespace(model="test", embed=AsyncMock(return_value=[(1.0,)]))
    result = asyncio.run(
        AskDisclosureHandler(repo, embedding, SimpleNamespace()).retrieve(
            ["005930"], question, from_date, None, True
        )
    )
    assert len(result) == expected
    assert result[0].filing.receipt_number == latest.receipt_number
