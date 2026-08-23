import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.tax.service import (
    TaxDocumentFields,
    TaxDocumentIssue,
    TaxDocumentService,
    _StructuredTaxVerification,
)


@pytest.mark.anyio
async def test_tax_document_uses_stateless_structured_file_input() -> None:
    parsed = _StructuredTaxVerification(
        detected_document_type="RESIDENCY_CERTIFICATE",
        verification_status="VERIFIED",
        fields=TaxDocumentFields(
            holder_name="Jane Investor",
            residency_country="US",
            issue_date="2026-01-10",
            issuing_authority="IRS",
        ),
        missing_required_fields=(),
        issues=(
            TaxDocumentIssue(
                code="AUTHENTICITY_NOT_CONFIRMED",
                severity="INFO",
                message="This screening does not constitute government approval.",
            ),
        ),
        ocr_confidence=0.97,
        tamper_risk=0.02,
        manual_review_required=False,
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=AsyncMock(return_value=SimpleNamespace(output_parsed=parsed))
        )
    )
    service = TaxDocumentService(client, Settings())
    content = base64.b64encode(b"%PDF-1.7\n%%EOF").decode()

    result = await service.verify(
        "RESIDENCY_CERTIFICATE",
        "certificate.pdf",
        "application/pdf",
        content,
        "US",
        "INDIVIDUAL",
        "a" * 64,
    )

    assert result.verification_status == "VERIFIED"
    kwargs = client.responses.parse.await_args.kwargs
    assert kwargs["store"] is False
    assert kwargs["safety_identifier"] == "a" * 64
    assert kwargs["input"][0]["content"][1]["type"] == "input_file"


@pytest.mark.anyio
async def test_tax_document_rejects_invalid_base64_before_provider_call() -> None:
    client = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock()))
    service = TaxDocumentService(client, Settings())

    with pytest.raises(AppError) as error:
        await service.verify(
            "APOSTILLE",
            "apostille.png",
            "image/png",
            "not-base64!",
            "US",
            "INDIVIDUAL",
            "b" * 64,
        )

    assert error.value.code == "INVALID_DOCUMENT"
    client.responses.parse.assert_not_awaited()
