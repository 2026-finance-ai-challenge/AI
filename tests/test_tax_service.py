import base64

import pytest

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.tax.document_model.schemas import DocumentType, ExtractedDocument
from k_market_ai.tax.service import (
    TaxBundleDocument,
    TaxDocumentService,
    _PipelineDocument,
)


def residency_document() -> _PipelineDocument:
    return _PipelineDocument(
        api_type="RESIDENCY_CERTIFICATE",
        extracted=ExtractedDocument(
            document_type=DocumentType.RESIDENCY_CERTIFICATE,
            source_path="certificate.png",
            fields={
                "taxpayer_name": "MARIA L. CHEN",
                "tin": "987-65-4321",
                "tax_year": "2026",
                "issue_date": "January 12, 2026",
                "residency_country": "United States of America",
                "residency_country_code": "US",
            },
            quality_checks={
                "has_certification_text": True,
                "has_irs_heading": True,
                "seal_present": True,
                "signature_present": True,
            },
        ),
        confidence=0.97,
        expected_country="US",
        investor_type="INDIVIDUAL",
    )


@pytest.mark.anyio
async def test_tax_document_uses_ported_document_model(monkeypatch: pytest.MonkeyPatch) -> None:
    service = TaxDocumentService(Settings())
    received: list[tuple[object, ...]] = []

    def process(*args: object) -> _PipelineDocument:
        received.append(args)
        return residency_document()

    monkeypatch.setattr(service, "_process_document", process)
    content = base64.b64encode(b"document").decode()

    result = await service.verify(
        "RESIDENCY_CERTIFICATE",
        "certificate.png",
        "image/png",
        content,
        "US",
        "INDIVIDUAL",
        "a" * 64,
    )

    assert result.verification_status == "VERIFIED"
    assert result.model == "kmarket-tax-document-ocr-runtime-v2"
    assert result.fields.holder_name == "MARIA L. CHEN"
    assert result.fields.issue_date == "2026-01-12"
    assert result.fields.residency_country == "US"
    assert result.tamper_risk == pytest.approx(0.03)
    assert received[0][3] == b"document"


@pytest.mark.anyio
async def test_tax_document_rejects_invalid_base64_before_ocr() -> None:
    service = TaxDocumentService(Settings())

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


@pytest.mark.anyio
async def test_bundle_requires_one_of_each_document_type() -> None:
    service = TaxDocumentService(Settings())
    encoded = base64.b64encode(b"document").decode()
    documents = tuple(
        TaxBundleDocument(
            document_type="RESIDENCY_CERTIFICATE",
            file_name=f"certificate-{index}.png",
            content_type="image/png",
            document_base64=encoded,
        )
        for index in range(3)
    )

    with pytest.raises(AppError) as error:
        await service.verify_bundle(documents, "US", "INDIVIDUAL", "c" * 64)

    assert error.value.code == "INVALID_TAX_DOCUMENT_BUNDLE"


@pytest.mark.anyio
async def test_bundle_runs_original_cross_document_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TaxDocumentService(Settings())
    encoded = base64.b64encode(b"document").decode()
    extracted = {
        "RESIDENCY_CERTIFICATE": residency_document(),
        "APOSTILLE": _PipelineDocument(
            api_type="APOSTILLE",
            extracted=ExtractedDocument(
                document_type=DocumentType.APOSTILLE,
                source_path="apostille.png",
                fields={
                    "issuing_country": "United States of America",
                    "signed_by": "CHONG U CHOI",
                    "signer_capacity": "NOTARY PUBLIC",
                    "seal_owner": "COUNTY OF MECKLENBURG, NORTH CAROLINA",
                    "issued_at": "Raleigh, North Carolina",
                    "issued_on": "10th April 2014",
                    "issuing_authority": "Secretary of State",
                    "certificate_number": "1185973223",
                },
                quality_checks={
                    "has_apostille_heading": True,
                    "seal_present": True,
                    "signature_present": True,
                },
            ),
            confidence=0.94,
            expected_country="US",
            investor_type="INDIVIDUAL",
        ),
        "REDUCED_TAX_APPLICATION": _PipelineDocument(
            api_type="REDUCED_TAX_APPLICATION",
            extracted=ExtractedDocument(
                document_type=DocumentType.WITHHOLDING_TAX_FORM,
                source_path="application.png",
                fields={
                    "first_name": "MARIA",
                    "middle_name": "L",
                    "last_name": "CHEN",
                    "address": "1234 Sunset Blvd, Los Angeles",
                    "tin": "987-65-4321",
                    "residency_country": "United States of America",
                    "residency_country_code": "US",
                    "dividend_tax_rate": "15%",
                    "signature_date": "2026-01-12",
                },
                quality_checks={
                    "all_no_boxes_checked": True,
                    "signature_present": True,
                },
            ),
            confidence=0.96,
            expected_country="US",
            investor_type="INDIVIDUAL",
        ),
    }

    def process(document_type: str, *_: object) -> _PipelineDocument:
        return extracted[document_type]

    monkeypatch.setattr(service, "_process_document", process)
    documents = tuple(
        TaxBundleDocument(
            document_type=document_type,
            file_name=f"{document_type}.png",
            content_type="image/png",
            document_base64=encoded,
        )
        for document_type in (
            "RESIDENCY_CERTIFICATE",
            "APOSTILLE",
            "REDUCED_TAX_APPLICATION",
        )
    )

    result = await service.verify_bundle(documents, "US", "INDIVIDUAL", "d" * 64)

    assert result.verification_status == "VERIFIED"
    assert result.cross_check["matched"] is True
    assert len(result.documents) == 3
