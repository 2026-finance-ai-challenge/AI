from __future__ import annotations

import re

from k_market_ai.tax.document_model.normalization import canonicalize_name
from k_market_ai.tax.document_model.schemas import (
    DocumentType,
    ExtractedDocument,
    ReviewFinding,
    ReviewResult,
    ReviewStatus,
)


class TaxDocumentReviewer:
    """Business-rule reviewer for extracted OCR fields and cross-document consistency."""

    def review(self, documents: list[ExtractedDocument]) -> ReviewResult:
        findings: list[ReviewFinding] = []
        indexed = {document.document_type: document for document in documents}

        residency_doc = indexed.get(DocumentType.RESIDENCY_CERTIFICATE)
        if residency_doc:
            findings.extend(self._review_residency_certificate(residency_doc))

        apostille_doc = indexed.get(DocumentType.APOSTILLE)
        if apostille_doc:
            findings.extend(self._review_apostille(apostille_doc))

        withholding_doc = indexed.get(DocumentType.WITHHOLDING_TAX_FORM)
        if withholding_doc:
            findings.extend(self._review_withholding_tax_form(withholding_doc))

        cross_check = self._cross_check(residency_doc, withholding_doc)
        findings.extend(cross_check["findings"])

        severe_findings = [item for item in findings if item.code.startswith("required_")]
        status = (
            ReviewStatus.PASS if not severe_findings and not findings else ReviewStatus.NEEDS_REVIEW
        )
        if severe_findings:
            status = ReviewStatus.REJECT

        return ReviewResult(
            status=status,
            findings=findings,
            cross_check={key: value for key, value in cross_check.items() if key != "findings"},
        )

    def cross_check(
        self,
        residency_document: ExtractedDocument,
        withholding_document: ExtractedDocument,
    ) -> ReviewResult:
        """Run only the original cross-document rules against cached OCR results."""
        cross_check = self._cross_check(residency_document, withholding_document)
        findings = cross_check["findings"]
        status = ReviewStatus.PASS
        if any(item.code.startswith("required_") for item in findings):
            status = ReviewStatus.REJECT
        elif findings or cross_check["matched"] is not True:
            status = ReviewStatus.NEEDS_REVIEW
        return ReviewResult(
            status=status,
            findings=findings,
            cross_check={key: value for key, value in cross_check.items() if key != "findings"},
        )

    def _review_residency_certificate(self, document: ExtractedDocument) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        if not document.fields.get("taxpayer_name"):
            findings.append(
                self._finding(
                    "required_name_missing",
                    "Taxpayer name is missing.",
                    "taxpayer_name",
                )
            )
        if not self._is_ssn_or_ein(document.fields.get("tin")):
            findings.append(self._finding("required_tin_invalid", "TIN format is invalid.", "tin"))
        if not document.fields.get("tax_year"):
            findings.append(
                self._finding(
                    "required_tax_year_missing",
                    "Tax year is missing.",
                    "tax_year",
                )
            )
        if not document.fields.get("issue_date"):
            findings.append(
                self._finding(
                    "required_issue_date_missing",
                    "Issue date is missing.",
                    "issue_date",
                )
            )
        if document.fields.get("residency_country") != "United States of America":
            findings.append(
                self._finding(
                    "required_residency_country_invalid",
                    "Residency country must be United States of America.",
                    "residency_country",
                )
            )
        if document.fields.get("residency_country_code") != "US":
            findings.append(
                self._finding(
                    "required_residency_country_code_invalid",
                    "Residency country code must be US.",
                    "residency_country_code",
                )
            )
        if not document.quality_checks.get("has_certification_text"):
            findings.append(
                self._finding(
                    "required_certification_text_missing",
                    "Certification text is missing.",
                )
            )
        if document.quality_checks.get("seal_present") is not True:
            findings.append(
                self._finding(
                    "required_certification_seal_missing",
                    "Certification seal was not detected.",
                    "seal_present",
                )
            )
        if document.quality_checks.get("signature_present") is not True:
            findings.append(
                self._finding(
                    "required_signature_missing",
                    "Signature was not detected.",
                    "signature_present",
                )
            )
        return findings

    def _review_apostille(self, document: ExtractedDocument) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        if not document.quality_checks.get("has_apostille_heading"):
            findings.append(
                self._finding(
                    "required_apostille_heading_missing",
                    "Apostille heading not found.",
                )
            )
        for field_name in [
            "issuing_country",
            "signed_by",
            "signer_capacity",
            "seal_owner",
            "issued_at",
            "issued_on",
            "issuing_authority",
            "certificate_number",
        ]:
            if not document.fields.get(field_name):
                findings.append(
                    self._finding(
                        "required_apostille_field_missing",
                        f"Apostille field {field_name} is missing.",
                        field_name,
                    )
                )
        if not self._looks_like_apostille_date(document.fields.get("issued_on")):
            findings.append(
                self._finding(
                    "required_apostille_date_invalid",
                    "Apostille date is incomplete or invalid.",
                    "issued_on",
                )
            )
        if document.quality_checks.get("seal_present") is not True:
            findings.append(
                self._finding(
                    "required_seal_missing",
                    "Apostille seal/stamp was not detected.",
                    "seal_present",
                )
            )
        if document.quality_checks.get("signature_present") is not True:
            findings.append(
                self._finding(
                    "required_signature_missing",
                    "Apostille signature was not detected.",
                    "signature_present",
                )
            )
        return findings

    def _review_withholding_tax_form(self, document: ExtractedDocument) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        if not document.fields.get("first_name") or not document.fields.get("last_name"):
            findings.append(self._finding("required_name_missing", "First/last name is missing."))
        if not document.fields.get("address"):
            findings.append(
                self._finding(
                    "required_address_missing",
                    "Address is missing.",
                    "address",
                )
            )
        if not self._is_ssn_or_ein(document.fields.get("tin")):
            findings.append(self._finding("required_tin_invalid", "TIN format is invalid.", "tin"))
        if document.fields.get("residency_country") not in {"United States of America", "USA"}:
            findings.append(
                self._finding(
                    "required_residency_country_invalid",
                    "Residency country must be United States of America.",
                    "residency_country",
                )
            )
        if document.fields.get("residency_country_code") != "US":
            findings.append(
                self._finding(
                    "required_residency_country_code_invalid",
                    "Residency country code must be US.",
                    "residency_country_code",
                )
            )
        if document.fields.get("dividend_tax_rate") != "15%":
            findings.append(
                self._finding(
                    "required_dividend_rate_invalid",
                    "Dividend tax rate must be 15%.",
                    "dividend_tax_rate",
                )
            )
        if not document.fields.get("signature_date"):
            findings.append(
                self._finding(
                    "required_signature_date_missing",
                    "Signature date is missing.",
                    "signature_date",
                )
            )
        if document.quality_checks.get("all_no_boxes_checked") is not True:
            findings.append(
                self._finding(
                    "required_no_checkbox_missing",
                    "All 'No' checkboxes must be checked.",
                    "all_no_boxes_checked",
                )
            )
        if document.quality_checks.get("signature_present") is not True:
            findings.append(
                self._finding(
                    "required_signature_missing",
                    "Signature was not detected.",
                    "signature_present",
                )
            )
        return findings

    def _cross_check(
        self,
        residency_doc: ExtractedDocument | None,
        withholding_doc: ExtractedDocument | None,
    ) -> dict[str, object]:
        findings: list[ReviewFinding] = []
        matched = True

        if not residency_doc or not withholding_doc:
            return {
                "matched": False,
                "reason": "cross-check skipped because one of the documents is missing",
                "findings": findings,
            }

        pairs = [
            ("tin", residency_doc.fields.get("tin"), withholding_doc.fields.get("tin")),
            (
                "residency_country",
                residency_doc.fields.get("residency_country"),
                withholding_doc.fields.get("residency_country"),
            ),
            (
                "residency_country_code",
                residency_doc.fields.get("residency_country_code"),
                withholding_doc.fields.get("residency_country_code"),
            ),
        ]
        for field_name, left_value, right_value in pairs:
            if left_value and right_value and str(left_value).strip() != str(right_value).strip():
                matched = False
                findings.append(
                    self._finding(
                        "required_cross_check_mismatch",
                        f"Cross-check mismatch for {field_name}.",
                        field_name,
                    )
                )

        residency_name = residency_doc.fields.get("taxpayer_name")
        withholding_name_parts = [
            withholding_doc.fields.get("first_name"),
            withholding_doc.fields.get("middle_name"),
            withholding_doc.fields.get("last_name"),
        ]
        withholding_name = " ".join(part for part in withholding_name_parts if part).strip()
        if (
            residency_name
            and withholding_name
            and canonicalize_name(residency_name) != canonicalize_name(withholding_name)
        ):
            matched = False
            findings.append(
                self._finding(
                    "required_cross_check_mismatch",
                    "Cross-check mismatch for taxpayer name.",
                    "taxpayer_name",
                )
            )

        return {
            "matched": matched,
            "reason": None if matched else "one or more cross-check fields do not match",
            "findings": findings,
        }

    @staticmethod
    def _is_ssn_or_ein(value: str | None) -> bool:
        if not value:
            return False
        normalized = value.strip()
        return bool(
            re.fullmatch(r"\d{3}-\d{2}-\d{4}", normalized)
            or re.fullmatch(r"\d{2}-\d{7}", normalized)
        )

    @staticmethod
    def _looks_like_apostille_date(value: str | None) -> bool:
        if not value:
            return False
        normalized = value.strip()
        return bool(
            re.search(r"\d{1,2}(?:st|nd|rd|th)?", normalized, re.IGNORECASE)
            and re.search(r"\d{4}", normalized)
            and re.search(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
                normalized,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _finding(code: str, message: str, field_name: str | None = None) -> ReviewFinding:
        return ReviewFinding(code=code, message=message, field_name=field_name)
