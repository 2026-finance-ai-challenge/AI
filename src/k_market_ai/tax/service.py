import asyncio
import base64
import binascii
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.tax.ocr import TaxOcrEngine, country_code, field_after, first_date


class TaxDocumentFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holder_name: str | None = Field(default=None, max_length=300)
    residency_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    issue_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    expiry_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    issuing_authority: str | None = Field(default=None, max_length=300)
    document_number: str | None = Field(default=None, max_length=200)
    apostille_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    treaty_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    investor_type: Literal["INDIVIDUAL", "CORPORATE"] | None = None


class TaxDocumentIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^[A-Z0-9_]{2,80}$")
    severity: Literal["INFO", "WARNING", "HIGH"]
    message: str = Field(min_length=1, max_length=500)


@dataclass(frozen=True, slots=True)
class TaxVerificationResult:
    detected_document_type: str
    verification_status: str
    fields: TaxDocumentFields
    missing_required_fields: tuple[str, ...]
    issues: tuple[TaxDocumentIssue, ...]
    ocr_confidence: float
    tamper_risk: float
    manual_review_required: bool
    model: str
    prompt_version: str


class TaxDocumentService:
    def __init__(self, settings: Settings, ocr_engine: TaxOcrEngine | None = None) -> None:
        self._ocr_engine = ocr_engine or TaxOcrEngine()
        self._model = "kmarket-tax-ocr-tesseract-rules-v1"
        self._prompt_version = settings.tax_document_prompt_version

    async def verify(
        self,
        document_type: str,
        file_name: str,
        content_type: str,
        document_base64: str,
        expected_residency_country: str,
        investor_type: str,
        safety_identifier: str,
    ) -> TaxVerificationResult:
        del safety_identifier
        content = self._decode_file(document_base64)
        ocr = await asyncio.to_thread(
            self._ocr_engine.read,
            content,
            content_type,
            file_name,
        )
        parsed = _review_document(
            document_type,
            ocr.text,
            ocr.confidence,
            expected_residency_country,
            investor_type,
        )
        return TaxVerificationResult(
            detected_document_type=parsed.detected_document_type,
            verification_status=parsed.verification_status,
            fields=parsed.fields,
            missing_required_fields=parsed.missing_required_fields,
            issues=parsed.issues,
            ocr_confidence=parsed.ocr_confidence,
            tamper_risk=parsed.tamper_risk,
            manual_review_required=parsed.manual_review_required,
            model=self._model,
            prompt_version=self._prompt_version,
        )

    @staticmethod
    def _decode_file(document_base64: str) -> bytes:
        try:
            content = base64.b64decode(document_base64, validate=True)
        except (binascii.Error, ValueError) as exception:
            raise AppError(
                code="INVALID_DOCUMENT",
                message="The document payload is invalid.",
                status_code=400,
            ) from exception
        if not content or len(content) > 10 * 1024 * 1024:
            raise AppError(
                code="INVALID_DOCUMENT",
                message="The document payload is invalid.",
                status_code=400,
            )
        return content


@dataclass(frozen=True, slots=True)
class _ReviewedDocument:
    detected_document_type: str
    verification_status: str
    fields: TaxDocumentFields
    missing_required_fields: tuple[str, ...]
    issues: tuple[TaxDocumentIssue, ...]
    ocr_confidence: float
    tamper_risk: float
    manual_review_required: bool


def _review_document(
    expected_type: str,
    text: str,
    confidence: float,
    expected_country: str,
    investor_type: str,
) -> _ReviewedDocument:
    detected = _detect_document_type(text)
    country = country_code(text, expected_country)
    normalized_investor: Literal["INDIVIDUAL", "CORPORATE"] = (
        "CORPORATE" if investor_type == "CORPORATE" else "INDIVIDUAL"
    )
    fields = TaxDocumentFields(
        holder_name=field_after(text, ("Taxpayer name", "Holder name", "Name", "성명")),
        residency_country=country,
        issue_date=first_date(text),
        expiry_date=None,
        issuing_authority=field_after(
            text,
            ("Issuing authority", "Competent authority", "Authority", "발급기관"),
        ),
        document_number=field_after(
            text,
            ("Certificate number", "Document number", "No.", "문서번호"),
        ),
        apostille_country=country if detected == "APOSTILLE" else None,
        treaty_country=country if detected == "REDUCED_TAX_APPLICATION" else None,
        investor_type=normalized_investor if detected == "REDUCED_TAX_APPLICATION" else None,
    )
    required = _required_fields(expected_type)
    missing = tuple(name for name in required if getattr(fields, name) is None)
    issues: list[TaxDocumentIssue] = []
    if detected != expected_type:
        issues.append(
            TaxDocumentIssue(
                code="DOCUMENT_TYPE_MISMATCH",
                severity="HIGH",
                message="The visible document type does not match the selected document type.",
            )
        )
    if country is None:
        issues.append(
            TaxDocumentIssue(
                code="RESIDENCY_COUNTRY_NOT_VERIFIED",
                severity="HIGH",
                message="The selected residency country was not found in the OCR text.",
            )
        )
    if confidence < 0.75:
        issues.append(
            TaxDocumentIssue(
                code="LOW_OCR_CONFIDENCE",
                severity="WARNING",
                message="OCR confidence is below the automatic-verification threshold.",
            )
        )
    if missing:
        issues.append(
            TaxDocumentIssue(
                code="REQUIRED_FIELDS_MISSING",
                severity="WARNING",
                message="One or more required document fields could not be read.",
            )
        )
    issues.append(
        TaxDocumentIssue(
            code="AUTHENTICITY_NOT_CONFIRMED",
            severity="INFO",
            message="OCR screening does not constitute government authenticity approval.",
        )
    )
    rejected = detected != expected_type or country is None
    verified = not rejected and confidence >= 0.75 and not missing
    status = "VERIFIED" if verified else "REJECTED" if rejected else "REVIEW_REQUIRED"
    quality_risk = round(max(0.0, min(1.0, 1.0 - confidence)), 4)
    return _ReviewedDocument(
        detected_document_type=detected,
        verification_status=status,
        fields=fields,
        missing_required_fields=missing,
        issues=tuple(issues),
        ocr_confidence=confidence,
        # 현재 API 필드명은 유지하되 OCR 품질 위험만 표시하고 진위 판정으로 사용하지 않는다.
        tamper_risk=quality_risk,
        manual_review_required=not verified,
    )


def _detect_document_type(text: str) -> str:
    normalized = text.upper()
    if "APOSTILLE" in normalized and "CONVENTION DE LA HAYE" in normalized:
        return "APOSTILLE"
    reduced_rate_tokens = (
        "APPLICATION FOR REDUCED TAX RATE",
        "LIMITED TAX RATE",
        "제한세율",
    )
    if any(token in normalized for token in reduced_rate_tokens):
        return "REDUCED_TAX_APPLICATION"
    residence_tokens = (
        "CERTIFICATE OF RESIDENCE",
        "RESIDENCY CERTIFICATE",
        "거주자증명서",
    )
    if any(token in normalized for token in residence_tokens):
        return "RESIDENCY_CERTIFICATE"
    return "UNKNOWN"


def _required_fields(document_type: str) -> tuple[str, ...]:
    if document_type == "APOSTILLE":
        return ("apostille_country", "issue_date", "issuing_authority", "document_number")
    if document_type == "REDUCED_TAX_APPLICATION":
        return ("holder_name", "treaty_country", "issue_date", "investor_type")
    return ("holder_name", "residency_country", "issue_date", "issuing_authority")
