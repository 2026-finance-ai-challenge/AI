import base64

import pytest

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.tax.ocr import OcrResult, _parse_tsv, _validated_suffix
from k_market_ai.tax.service import TaxDocumentService, _review_document


class FakeOcrEngine:
    def __init__(self) -> None:
        self.received: tuple[bytes, str, str] | None = None

    def read(self, content: bytes, content_type: str, file_name: str) -> OcrResult:
        self.received = (content, content_type, file_name)
        return OcrResult(
            text="""Certificate of Residence
Taxpayer name: Jane Investor
Country: United States
Issue date: 2026-01-10
Issuing authority: Internal Revenue Service""",
            confidence=0.97,
        )


@pytest.mark.anyio
async def test_tax_document_uses_local_ocr_and_rule_review() -> None:
    engine = FakeOcrEngine()
    service = TaxDocumentService(Settings(), engine)
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
    assert result.model == "kmarket-tax-ocr-tesseract-rules-v1"
    assert result.fields.residency_country == "US"
    assert result.tamper_risk == pytest.approx(0.03)
    assert engine.received == (b"%PDF-1.7\n%%EOF", "application/pdf", "certificate.pdf")


@pytest.mark.anyio
async def test_tax_document_rejects_invalid_base64_before_provider_call() -> None:
    engine = FakeOcrEngine()
    service = TaxDocumentService(Settings(), engine)

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
    assert engine.received is None


def test_document_signature_and_extension_must_both_match() -> None:
    assert _validated_suffix(b"\x89PNG\r\n\x1a\nbody", "image/png", "scan.png") == ".png"

    with pytest.raises(AppError) as error:
        _validated_suffix(b"\x89PNG\r\n\x1a\nbody", "image/png", "scan.pdf")

    assert error.value.code == "INVALID_DOCUMENT"


def test_tsv_parser_preserves_visual_lines_and_ignores_invalid_confidence() -> None:
    payload = "\t".join(("page_num", "block_num", "par_num", "line_num", "conf", "text"))
    payload += "\n" + "\t".join(("1", "1", "1", "1", "96.5", "Certificate"))
    payload += "\n" + "\t".join(("1", "1", "1", "1", "invalid", "of Residence"))
    payload += "\n" + "\t".join(("1", "1", "1", "2", "88", "Name: Jane"))

    lines, confidence = _parse_tsv(payload)

    assert lines == ["Certificate of Residence", "Name: Jane"]
    assert confidence == [96.5, 88.0]


def test_rule_review_rejects_document_type_mismatch_without_fallback() -> None:
    reviewed = _review_document(
        "APOSTILLE",
        """Certificate of Residence
Taxpayer name: Jane Investor
Country: United States
Issue date: 2026-01-10
Issuing authority: Internal Revenue Service""",
        0.98,
        "US",
        "INDIVIDUAL",
    )

    assert reviewed.verification_status == "REJECTED"
    assert reviewed.detected_document_type == "RESIDENCY_CERTIFICATE"
    assert reviewed.manual_review_required is True
    assert "DOCUMENT_TYPE_MISMATCH" in {issue.code for issue in reviewed.issues}
