from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.core.config import Settings
from k_market_ai.core.errors import AppError
from k_market_ai.tax.document_model.ocr import TesseractOCREngine
from k_market_ai.tax.document_model.pipeline import TaxDocumentPipeline
from k_market_ai.tax.document_model.review import TaxDocumentReviewer
from k_market_ai.tax.document_model.schemas import (
    DocumentType,
    ExtractedDocument,
    ReviewFinding,
    ReviewResult,
    ReviewStatus,
)

ApiDocumentType = Literal["RESIDENCY_CERTIFICATE", "APOSTILLE", "REDUCED_TAX_APPLICATION"]


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
    birth_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    phone_number: str | None = Field(default=None, max_length=80)
    address: str | None = Field(default=None, max_length=1000)
    preview_version: int | None = Field(default=None, ge=0, le=1)


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


@dataclass(frozen=True, slots=True)
class TaxBundleDocument:
    document_type: ApiDocumentType
    file_name: str
    content_type: str
    document_base64: str


@dataclass(frozen=True, slots=True)
class TaxBundleResult:
    verification_status: str
    findings: tuple[TaxDocumentIssue, ...]
    cross_check: dict[str, object]
    documents: tuple[TaxVerificationResult, ...]
    model: str


@dataclass(frozen=True, slots=True)
class TaxCachedDocument:
    document_type: ApiDocumentType
    detected_document_type: ApiDocumentType
    verification_status: str
    fields: TaxDocumentFields
    missing_required_fields: tuple[str, ...]
    issues: tuple[TaxDocumentIssue, ...]
    ocr_confidence: float
    tamper_risk: float
    manual_review_required: bool
    model: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class _PipelineDocument:
    api_type: ApiDocumentType
    extracted: ExtractedDocument
    confidence: float
    expected_country: str
    investor_type: str


class TaxDocumentService:
    def __init__(self, settings: Settings) -> None:
        self._model = "kmarket-tax-document-ocr-runtime-v2"
        self._prompt_version = settings.tax_document_prompt_version

    async def verify(
        self,
        document_type: ApiDocumentType,
        file_name: str,
        content_type: str,
        document_base64: str,
        expected_residency_country: str,
        investor_type: str,
        safety_identifier: str,
    ) -> TaxVerificationResult:
        del safety_identifier
        pipeline_document = await asyncio.to_thread(
            self._process_document,
            document_type,
            file_name,
            content_type,
            self._decode_file(document_base64),
            expected_residency_country,
            investor_type,
        )
        review = TaxDocumentReviewer().review([pipeline_document.extracted])
        return self._to_verification(pipeline_document, review)

    async def verify_bundle(
        self,
        documents: tuple[TaxBundleDocument, ...],
        expected_residency_country: str,
        investor_type: str,
        safety_identifier: str,
    ) -> TaxBundleResult:
        del safety_identifier
        required = {
            "RESIDENCY_CERTIFICATE",
            "APOSTILLE",
            "REDUCED_TAX_APPLICATION",
        }
        supplied = {document.document_type for document in documents}
        if len(documents) != 3 or supplied != required:
            raise AppError(
                code="INVALID_TAX_DOCUMENT_BUNDLE",
                message="Exactly one document of each required tax-document type is required.",
                status_code=400,
            )
        processed = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._process_document,
                    document.document_type,
                    document.file_name,
                    document.content_type,
                    self._decode_file(document.document_base64),
                    expected_residency_country,
                    investor_type,
                )
                for document in documents
            )
        )
        review = TaxDocumentReviewer().review([item.extracted for item in processed])
        per_document = tuple(
            self._to_verification(item, TaxDocumentReviewer().review([item.extracted]))
            for item in processed
        )
        bundle_status = _review_status(review.status)
        if bundle_status == "VERIFIED" and any(
            document.verification_status != "VERIFIED" for document in per_document
        ):
            bundle_status = "REVIEW_REQUIRED"
        return TaxBundleResult(
            verification_status=bundle_status,
            findings=tuple(_issue(finding) for finding in review.findings),
            cross_check=review.cross_check,
            documents=per_document,
            model=self._model,
        )

    async def compare_cached(
        self,
        documents: tuple[TaxCachedDocument, ...],
        expected_residency_country: str,
        investor_type: str,
        safety_identifier: str,
    ) -> TaxBundleResult:
        del safety_identifier
        required = {
            "RESIDENCY_CERTIFICATE",
            "APOSTILLE",
            "REDUCED_TAX_APPLICATION",
        }
        supplied = {document.document_type for document in documents}
        detected = {document.detected_document_type for document in documents}
        if len(documents) != 3 or supplied != required or detected != required:
            raise AppError(
                code="INVALID_TAX_DOCUMENT_BUNDLE",
                message="Exactly one verified result of each required type is required.",
                status_code=400,
            )
        indexed = {document.document_type: document for document in documents}
        residency = indexed["RESIDENCY_CERTIFICATE"]
        withholding = indexed["REDUCED_TAX_APPLICATION"]
        if residency.fields.residency_country not in {None, expected_residency_country}:
            raise AppError(
                code="INVALID_TAX_DOCUMENT_BUNDLE",
                message="The cached residence country does not match the request.",
                status_code=400,
            )
        if withholding.fields.treaty_country not in {None, expected_residency_country}:
            raise AppError(
                code="INVALID_TAX_DOCUMENT_BUNDLE",
                message="The cached treaty country does not match the request.",
                status_code=400,
            )
        if withholding.fields.investor_type not in {None, investor_type}:
            raise AppError(
                code="INVALID_TAX_DOCUMENT_BUNDLE",
                message="The cached investor type does not match the request.",
                status_code=400,
            )
        review = TaxDocumentReviewer().cross_check(
            _cached_extraction(residency),
            _cached_extraction(withholding),
        )
        status = _cached_bundle_status(documents, review)
        return TaxBundleResult(
            verification_status=status,
            findings=tuple(_issue(finding) for finding in review.findings),
            cross_check=review.cross_check,
            documents=tuple(_cached_verification(document) for document in documents),
            model=self._model,
        )

    def _process_document(
        self,
        document_type: ApiDocumentType,
        file_name: str,
        content_type: str,
        content: bytes,
        expected_country: str,
        investor_type: str,
    ) -> _PipelineDocument:
        model_type = _model_type(document_type)
        started = time.monotonic()
        suffix = _safe_suffix(file_name, content_type)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="kmarket-tax-ocr-", suffix=suffix, delete=False
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            engine = TesseractOCREngine(
                lang="kor+eng" if model_type == DocumentType.WITHHOLDING_TAX_FORM else "eng"
            )
            result = TaxDocumentPipeline(ocr_engine=engine).process(
                model_type,
                temporary_path,
                source_name=file_name,
            )
            return _PipelineDocument(
                api_type=document_type,
                extracted=result.extracted_document,
                confidence=result.ocr_confidence,
                expected_country=expected_country,
                investor_type=investor_type,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exception:
            raise AppError(
                code="TAX_OCR_FAILED",
                message="The tax document could not be read by the OCR model.",
                status_code=422,
            ) from exception
        finally:
            logging.getLogger(__name__).info(
                "Tax OCR finished type=%s seconds=%.2f", document_type, time.monotonic() - started
            )
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _to_verification(
        self,
        item: _PipelineDocument,
        review: ReviewResult,
    ) -> TaxVerificationResult:
        fields = _fields(item)
        missing = tuple(_missing_fields(item.api_type, fields))
        issues = [_issue(finding) for finding in review.findings]
        if _document_country(fields, item.api_type) not in {None, item.expected_country}:
            issues.append(
                TaxDocumentIssue(
                    code="RESIDENCY_COUNTRY_MISMATCH",
                    severity="HIGH",
                    message="The document country does not match the selected tax residence.",
                )
            )
        status = _review_status(review.status)
        if missing and status == "VERIFIED":
            status = "REVIEW_REQUIRED"
        confidence = max(0.0, min(1.0, item.confidence))
        return TaxVerificationResult(
            detected_document_type=item.api_type,
            verification_status=status,
            fields=fields,
            missing_required_fields=missing,
            issues=tuple(issues),
            ocr_confidence=round(confidence, 4),
            # 이 필드는 진위 확률이 아니라 OCR 품질 위험을 나타낸다.
            tamper_risk=round(1.0 - confidence, 4),
            manual_review_required=status != "VERIFIED",
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


def _model_type(document_type: ApiDocumentType) -> DocumentType:
    return {
        "RESIDENCY_CERTIFICATE": DocumentType.RESIDENCY_CERTIFICATE,
        "APOSTILLE": DocumentType.APOSTILLE,
        "REDUCED_TAX_APPLICATION": DocumentType.WITHHOLDING_TAX_FORM,
    }[document_type]


def _cached_extraction(document: TaxCachedDocument) -> ExtractedDocument:
    fields = document.fields
    if document.document_type == "RESIDENCY_CERTIFICATE":
        source = {
            "taxpayer_name": fields.holder_name,
            "tin": fields.document_number,
            "residency_country": _country_name(fields.residency_country),
            "residency_country_code": fields.residency_country,
        }
        document_type = DocumentType.RESIDENCY_CERTIFICATE
    else:
        source = {
            "first_name": fields.holder_name,
            "middle_name": None,
            "last_name": None,
            "tin": fields.document_number,
            "residency_country": _country_name(fields.treaty_country),
            "residency_country_code": fields.treaty_country,
        }
        document_type = DocumentType.WITHHOLDING_TAX_FORM
    return ExtractedDocument(
        document_type=document_type,
        source_path="cached-verification-result",
        fields=source,
    )


def _country_name(country_code: str | None) -> str | None:
    return "United States of America" if country_code == "US" else country_code


def _cached_bundle_status(
    documents: tuple[TaxCachedDocument, ...],
    review: ReviewResult,
) -> str:
    statuses = {document.verification_status for document in documents}
    if "REJECTED" in statuses or review.status == ReviewStatus.REJECT:
        return "REJECTED"
    if statuses != {"VERIFIED"} or review.status != ReviewStatus.PASS:
        return "REVIEW_REQUIRED"
    return "VERIFIED"


def _cached_verification(document: TaxCachedDocument) -> TaxVerificationResult:
    return TaxVerificationResult(
        detected_document_type=document.detected_document_type,
        verification_status=document.verification_status,
        fields=document.fields,
        missing_required_fields=document.missing_required_fields,
        issues=document.issues,
        ocr_confidence=document.ocr_confidence,
        tamper_risk=document.tamper_risk,
        manual_review_required=document.manual_review_required,
        model=document.model,
        prompt_version=document.prompt_version,
    )


def _safe_suffix(file_name: str, content_type: str) -> str:
    content_suffix = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
    }.get(content_type)
    suffix = Path(file_name).suffix.lower()
    if content_suffix is None or suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise AppError(
            code="INVALID_DOCUMENT",
            message="The document media type is not supported.",
            status_code=400,
        )
    return content_suffix


def _fields(item: _PipelineDocument) -> TaxDocumentFields:
    source = item.extracted.fields
    if item.api_type == "RESIDENCY_CERTIFICATE":
        return TaxDocumentFields(
            holder_name=_text(source.get("taxpayer_name"), 300),
            residency_country=_country(source.get("residency_country_code")),
            issue_date=_date(source.get("issue_date")),
            issuing_authority=(
                "Internal Revenue Service"
                if item.extracted.quality_checks.get("has_irs_heading")
                else None
            ),
            document_number=_text(source.get("tin"), 200),
        )
    if item.api_type == "APOSTILLE":
        return TaxDocumentFields(
            holder_name=_text(source.get("signed_by"), 300),
            issue_date=_date(source.get("issued_on")),
            issuing_authority=_text(source.get("issuing_authority"), 300),
            document_number=_text(source.get("certificate_number"), 200),
            apostille_country=_country(source.get("issuing_country")) or "US",
        )
    holder = " ".join(
        str(source.get(key) or "").strip() for key in ("first_name", "middle_name", "last_name")
    ).strip()
    normalized_investor: Literal["INDIVIDUAL", "CORPORATE"] = (
        "CORPORATE" if item.investor_type == "CORPORATE" else "INDIVIDUAL"
    )
    return TaxDocumentFields(
        holder_name=_text(holder, 300),
        issue_date=_date(source.get("signature_date")),
        document_number=_text(source.get("tin"), 200),
        treaty_country=_country(source.get("residency_country_code")),
        investor_type=normalized_investor,
        birth_date=_date(source.get("birth_date")),
        phone_number=_text(source.get("phone_number"), 80),
        address=_text(source.get("address"), 1000),
        preview_version=1,
    )


def _country(value: object) -> str | None:
    normalized = str(value or "").strip().upper()
    if normalized in {"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}:
        return "US"
    return normalized if re.fullmatch(r"[A-Z]{2}", normalized) else None


def _date(value: object) -> str | None:
    normalized = str(value or "").strip().replace(".", " ")
    if not normalized:
        return None
    normalized = re.sub(r"(?i)(\d{1,2})(?:st|nd|rd|th)?\s+day\s+of\s+", r"\1 ", normalized)
    for pattern in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%B %d %Y",
        "%d %B %Y",
        "%d %B, %Y",
        "%dth %B %Y",
        "%dst %B %Y",
        "%dnd %B %Y",
        "%drd %B %Y",
    ):
        try:
            parsed = datetime.strptime(normalized, pattern).date()
            return parsed.isoformat()
        except ValueError:
            continue
    match = re.search(r"(20\d{2})\D+(\d{1,2})\D+(\d{1,2})", normalized)
    if match:
        try:
            return date(*(int(part) for part in match.groups())).isoformat()
        except ValueError:
            return None
    return None


def _text(value: object, maximum: int) -> str | None:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    return normalized[:maximum] or None


def _review_status(status: ReviewStatus) -> str:
    return {
        ReviewStatus.PASS: "VERIFIED",
        ReviewStatus.NEEDS_REVIEW: "REVIEW_REQUIRED",
        ReviewStatus.REJECT: "REJECTED",
    }[status]


def _issue(finding: ReviewFinding) -> TaxDocumentIssue:
    severity: Literal["INFO", "WARNING", "HIGH"] = (
        "HIGH" if finding.code.startswith("required_") else "WARNING"
    )
    code = re.sub(r"[^A-Z0-9_]", "_", finding.code.upper())[:80]
    return TaxDocumentIssue(code=code, severity=severity, message=finding.message[:500])


def _missing_fields(
    document_type: ApiDocumentType,
    fields: TaxDocumentFields,
) -> list[str]:
    required = {
        "RESIDENCY_CERTIFICATE": (
            "holder_name",
            "residency_country",
            "issue_date",
            "issuing_authority",
        ),
        "APOSTILLE": (
            "apostille_country",
            "issue_date",
            "issuing_authority",
            "document_number",
        ),
        "REDUCED_TAX_APPLICATION": (
            "holder_name",
            "treaty_country",
            "issue_date",
            "investor_type",
        ),
    }[document_type]
    return [name for name in required if getattr(fields, name) is None]


def _document_country(
    fields: TaxDocumentFields,
    document_type: ApiDocumentType,
) -> str | None:
    if document_type == "RESIDENCY_CERTIFICATE":
        return fields.residency_country
    if document_type == "APOSTILLE":
        return fields.apostille_country
    return fields.treaty_country
