from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    RESIDENCY_CERTIFICATE = "residency_certificate"
    APOSTILLE = "apostille"
    WITHHOLDING_TAX_FORM = "withholding_tax_form"


class ReviewStatus(StrEnum):
    PASS = "pass"  # noqa: S105  # nosec B105
    REJECT = "reject"
    NEEDS_REVIEW = "needs_review"


class OCRWordBox(BaseModel):
    text: str
    confidence: float | None = None
    points: list[list[float]] = Field(default_factory=list)


class OCRPage(BaseModel):
    page_number: int
    words: list[OCRWordBox] = Field(default_factory=list)
    raw_text: str = ""


class OCRResult(BaseModel):
    pages: list[OCRPage] = Field(default_factory=list)
    regions: dict[str, OCRPage] = Field(default_factory=dict)
    template_id: str | None = None

    def combined_text(self) -> str:
        return "\n".join(page.raw_text for page in self.pages if page.raw_text).strip()

    def region_text(self, region_name: str) -> str | None:
        page = self.regions.get(region_name)
        if page is None:
            return None
        return page.raw_text.strip() or None


class ExtractedDocument(BaseModel):
    document_type: DocumentType
    source_path: str
    template_id: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    quality_checks: dict[str, Any] = Field(default_factory=dict)
    parser_warnings: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    code: str
    message: str
    field_name: str | None = None


class ReviewResult(BaseModel):
    status: ReviewStatus
    findings: list[ReviewFinding] = Field(default_factory=list)
    cross_check: dict[str, Any] = Field(default_factory=dict)
