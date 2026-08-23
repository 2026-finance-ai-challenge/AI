from uuid import uuid4

from fastapi import FastAPI, Query
from fastapi.testclient import TestClient

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.main import create_app
from k_market_ai.news.domain import (
    MarketImpact,
    NewsAnalysis,
    NewsImportance,
    NewsSentiment,
    TermExplanation,
)
from k_market_ai.rag.application.disclosure_insight import DisclosureInsight
from k_market_ai.rag.domain.models import Citation, RagAnswer


def test_health() -> None:
    app = create_app(Settings(environment="test", allowed_hosts=["testserver"]))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_untrusted_host_is_rejected() -> None:
    app = create_app(Settings(environment="test", allowed_hosts=["testserver"]))

    with TestClient(app) as client:
        response = client.get("/health", headers={"host": "untrusted.example"})

    assert response.status_code == 400


def test_docs_are_disabled_in_production() -> None:
    app = create_app(Settings(environment="production", allowed_hosts=["testserver"]))

    with TestClient(app) as client:
        response = client.get("/docs")

    assert response.status_code == 404


def test_app_error_returns_safe_message() -> None:
    app = _test_app()

    @app.get("/test/app-error")
    async def raise_app_error() -> None:
        raise AppError(code="NOT_FOUND", message="The resource was not found.", status_code=404)

    with TestClient(app) as client:
        response = client.get("/test/app-error")

    assert response.status_code == 404
    assert response.json()["message"] == "The resource was not found."
    assert response.json()["request_id"]


def test_validation_error_does_not_echo_input() -> None:
    app = _test_app()

    @app.get("/test/validation")
    async def validate(limit: int = Query(ge=1, le=100)) -> dict[str, int]:
        return {"limit": limit}

    with TestClient(app) as client:
        response = client.get("/test/validation", params={"limit": "private-value"})

    body = response.json()
    assert response.status_code == 422
    assert body["code"] == "INVALID_REQUEST"
    assert "private-value" not in response.text


def test_unexpected_error_hides_internal_message() -> None:
    app = _test_app()

    @app.get("/test/unexpected")
    async def raise_unexpected_error() -> None:
        raise RuntimeError("database-password")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test/unexpected")

    assert response.status_code == 500
    assert response.json()["message"] == "An unexpected error occurred."
    assert "database-password" not in response.text


def test_rag_endpoint_requires_internal_service_token() -> None:
    service_token = str(uuid4())
    app = create_app(
        Settings(
            environment="test",
            allowed_hosts=["testserver"],
            service_token=service_token,
        )
    )

    with TestClient(app) as client:
        missing = client.post(
            "/internal/v1/disclosures/20260818800670/questions",
            json={"question": "What changed?"},
        )
        invalid = client.post(
            "/internal/v1/disclosures/20260818800670/questions",
            headers={"authorization": f"Bearer {service_token}-wrong"},
            json={"question": "What changed?"},
        )
        configured_without_handler = client.post(
            "/internal/v1/disclosures/20260818800670/questions",
            headers={"authorization": f"Bearer {service_token}"},
            json={"question": "What changed?"},
        )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert configured_without_handler.status_code == 503


def test_rag_endpoint_returns_english_answer_with_source_reference() -> None:
    service_token = str(uuid4())
    citation = Citation(
        id="C1",
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version=1,
        section_ids=(uuid4(),),
        first_ordinal=1,
        last_ordinal=2,
        heading="Revenue",
        excerpt="Revenue increased due to overseas demand.",
    )
    handler = FakeRagHandler(citation)
    app = create_app(
        Settings(
            environment="test",
            allowed_hosts=["testserver"],
            service_token=service_token,
        ),
        rag_handler=handler,
    )

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/disclosures/20260818800670/questions",
            headers={"authorization": f"Bearer {service_token}"},
            json={"question": "Why did revenue increase?"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Revenue increased due to overseas demand. [C1]"
    assert response.json()["citations"][0]["id"] == "C1"
    assert handler.receipt_number == "20260818800670"


def test_news_endpoints_require_service_token_and_return_structured_results() -> None:
    service_token = str(uuid4())
    service = FakeNewsService()
    app = create_app(
        Settings(
            environment="test",
            allowed_hosts=["testserver"],
            service_token=service_token,
        ),
        news_service=service,
    )

    with TestClient(app) as client:
        missing = client.post(
            "/internal/v1/news/analysis",
            json={"title": "제목", "paragraphs": ["본문"]},
        )
        analyzed = client.post(
            "/internal/v1/news/analysis",
            headers={"authorization": f"Bearer {service_token}"},
            json={"title": "제목", "paragraphs": ["본문"]},
        )
        explained = client.post(
            "/internal/v1/news/terms/explanations",
            headers={"authorization": f"Bearer {service_token}"},
            json={
                "selected_text": "유상증자",
                "article_context": "유상증자를 결정했다.",
                "evidence": [],
                "safety_identifier": "a" * 64,
            },
        )

    assert missing.status_code == 401
    assert analyzed.status_code == 200
    assert analyzed.json()["sentiment"] == "NEUTRAL"
    assert analyzed.json()["what"] == "A company announcement was reported."
    assert explained.status_code == 200
    assert explained.json()["sufficient_evidence"] is False
    assert service.safety_identifier == "a" * 64


def test_disclosure_insight_endpoint_requires_token_and_returns_citations() -> None:
    service_token = str(uuid4())
    service = FakeDisclosureInsightService()
    app = create_app(
        Settings(
            environment="test",
            allowed_hosts=["testserver"],
            service_token=service_token,
        ),
        disclosure_insight_service=service,
    )

    request_body = {
        "receipt_number": "20260823800001",
        "title": "Major Business Report",
        "evidence": [{"id": "S1", "heading": "Investment", "content": "New facility approved."}],
    }
    with TestClient(app) as client:
        missing = client.post("/internal/v1/disclosures/summaries", json=request_body)
        generated = client.post(
            "/internal/v1/disclosures/summaries",
            headers={"authorization": f"Bearer {service_token}"},
            json=request_body,
        )

    assert missing.status_code == 401
    assert generated.status_code == 200
    assert generated.json()["what"] == "A facility was approved."
    assert generated.json()["evidence_ids"] == ["S1"]
    assert service.receipt_number == "20260823800001"


class FakeRagHandler:
    def __init__(self, citation: Citation) -> None:
        self._citation = citation
        self.receipt_number: str | None = None

    async def ask(self, receipt_number: str, question: str, selected: object) -> RagAnswer:
        self.receipt_number = receipt_number
        assert question == "Why did revenue increase?"
        assert selected is None
        return RagAnswer(
            answer="Revenue increased due to overseas demand. [C1]",
            refused=False,
            refusal_reason=None,
            citations=(self._citation,),
            model="test-model",
            prompt_version="test-prompt",
        )


class FakeNewsService:
    def __init__(self) -> None:
        self.safety_identifier: str | None = None

    async def analyze(
        self,
        title: str,
        paragraphs: tuple[str, ...],
        candidate_companies: tuple[str, ...],
    ) -> NewsAnalysis:
        assert title == "제목"
        assert paragraphs == ("본문",)
        assert candidate_companies == ()
        return NewsAnalysis(
            english_title="Headline",
            translated_paragraphs=("Body",),
            what="A company announcement was reported.",
            why="The source does not state a reason.",
            impact="No impact is stated in the source.",
            event_type="COMPANY_UPDATE",
            sentiment=NewsSentiment.NEUTRAL,
            importance=NewsImportance.LOW,
            market_impact=MarketImpact.UNCERTAIN,
            event_confidence=0.8,
            sentiment_confidence=0.8,
            importance_confidence=0.7,
            market_impact_confidence=0.6,
            model="test-model",
            prompt_version="test-prompt",
        )

    async def explain_term(
        self,
        selected_text: str,
        article_context: str,
        evidence: tuple[object, ...],
        safety_identifier: str | None,
    ) -> TermExplanation:
        self.safety_identifier = safety_identifier
        assert selected_text == "유상증자"
        assert article_context == "유상증자를 결정했다."
        assert evidence == ()
        return TermExplanation(
            normalized_term=None,
            definition=None,
            contextual_meaning=None,
            evidence_ids=(),
            confidence=0.1,
            review_required=True,
            sufficient_evidence=False,
            refusal_reason="The supplied context is insufficient.",
            model="test-model",
            prompt_version="test-prompt",
        )


class FakeDisclosureInsightService:
    def __init__(self) -> None:
        self.receipt_number: str | None = None

    async def summarize(
        self,
        receipt_number: str,
        title: str,
        evidence: tuple[object, ...],
    ) -> DisclosureInsight:
        self.receipt_number = receipt_number
        assert title == "Major Business Report"
        assert len(evidence) == 1
        return DisclosureInsight(
            what="A facility was approved.",
            why="The filing states an expansion purpose.",
            impact="Capacity may increase.",
            evidence_ids=("S1",),
            sufficient_evidence=True,
            refusal_reason=None,
            model="test-model",
            prompt_version="filing-summary-test-v1",
        )


def _test_app() -> FastAPI:
    return create_app(Settings(environment="test", allowed_hosts=["testserver"]))
