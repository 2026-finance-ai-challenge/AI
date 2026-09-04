import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

from k_market_ai.core.config import Settings
from k_market_ai.main import create_app
from k_market_ai.rag.application.ask_disclosure import AskDisclosureHandler
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
    assert '["3개월", "누적", "3개월", "누적"]' in result[0].content
    assert '["매출액", "100", "200", "90", "180"]' in result[0].content
