from typing import Annotated, Literal, cast

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from k_market_ai.api.internal_auth import authenticate_internal
from k_market_ai.core.errors import AppError
from k_market_ai.tax.service import TaxDocumentFields, TaxDocumentIssue, TaxDocumentService

router = APIRouter(
    prefix="/internal/v1/tax",
    tags=["internal-tax"],
)


class TaxVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["RESIDENCY_CERTIFICATE", "APOSTILLE", "REDUCED_TAX_APPLICATION"]
    file_name: str = Field(min_length=1, max_length=255)
    content_type: Literal["application/pdf", "image/jpeg", "image/png"]
    document_base64: str = Field(min_length=4, max_length=14_000_000)
    expected_residency_country: str = Field(pattern=r"^[A-Z]{2}$")
    investor_type: Literal["INDIVIDUAL", "CORPORATE"]
    safety_identifier: str = Field(pattern=r"^[a-f0-9]{64}$")


class TaxVerificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


def service(request: Request) -> TaxDocumentService:
    value: object | None = getattr(request.app.state, "tax_document_service", None)
    if value is None:
        raise AppError(
            code="AI_NOT_CONFIGURED",
            message="Tax document verification is not configured.",
            status_code=503,
        )
    return cast(TaxDocumentService, value)


@router.post("/documents/verify", response_model=TaxVerificationResponse)
async def verify_document(
    body: TaxVerificationRequest,
    tax_service: Annotated[TaxDocumentService, Depends(service)],
    _: Annotated[None, Depends(authenticate_internal)],
) -> TaxVerificationResponse:
    result = await tax_service.verify(
        document_type=body.document_type,
        file_name=body.file_name,
        content_type=body.content_type,
        document_base64=body.document_base64,
        expected_residency_country=body.expected_residency_country,
        investor_type=body.investor_type,
        safety_identifier=body.safety_identifier,
    )
    return TaxVerificationResponse(
        detected_document_type=result.detected_document_type,
        verification_status=result.verification_status,
        fields=result.fields,
        missing_required_fields=result.missing_required_fields,
        issues=result.issues,
        ocr_confidence=result.ocr_confidence,
        tamper_risk=result.tamper_risk,
        manual_review_required=result.manual_review_required,
        model=result.model,
        prompt_version=result.prompt_version,
    )
