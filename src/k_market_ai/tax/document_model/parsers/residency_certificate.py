from __future__ import annotations

import re
from pathlib import Path

from k_market_ai.tax.document_model.normalization import normalize_country
from k_market_ai.tax.document_model.parsers.base import BaseDocumentParser
from k_market_ai.tax.document_model.schemas import DocumentType, ExtractedDocument, OCRResult


class ResidencyCertificateParser(BaseDocumentParser):
    document_type = DocumentType.RESIDENCY_CERTIFICATE

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        single_line = self._normalize_whitespace(text)
        tin_region_text = self._region_value(ocr_result, "tin")
        tin_source = self._normalize_whitespace(
            " ".join(
                part
                for part in [
                    tin_region_text,
                    single_line,
                ]
                if part
            )
        )
        region_taxpayer_raw = self._region_value(ocr_result, "taxpayer_name")
        region_taxpayer = self._clean_taxpayer_value(region_taxpayer_raw)
        # 단어별 줄바꿈 OCR에서는 전체 이름이 잘리지 않도록 단일 행 정규화 결과를 우선한다.
        fallback_taxpayer = (
            self._clean_taxpayer_value(
                self._find_first(
                    r"Taxpayer\s*[:;]?\s*(.+?)\s+(?:TIN|Tax Year|Date)\b",
                    single_line,
                )
            )
            or self._clean_taxpayer_value(self._find_first(r"Taxpayer\s*[:;]?\s*([^\n]+)", text))
            or self._clean_taxpayer_value(
                self._find_first(
                    r"Taxpayer\s*[:;]?\s*(.+?)\s+TIN\b",
                    self._normalize_whitespace(tin_region_text or ""),
                )
            )
        )
        taxpayer_name = self._normalize_name(
            self._select_taxpayer_name_candidate(
                region_taxpayer,
                fallback_taxpayer,
                region_taxpayer_raw=region_taxpayer_raw,
            )
        )
        tin = self._extract_first_pattern(
            [
                r"\bTIN\s*[:;]?\s*(\d{3}-\d{2}-\d{4})",
                r"\bTIN\s*[:;]?\s*(\d{2}-\d{7})",
                r"\b(\d{3}-\d{2}-\d{4})\b",
                r"\b(\d{2}-\d{7})\b",
            ],
            tin_source,
        )
        tax_year = self._find_first(
            r"(\d{4})", self._region_value(ocr_result, "tax_year") or ""
        ) or self._find_first(r"Tax\s*Year\s*[:;]?\s*(\d{4})", single_line)
        region_issue_date = self._normalize_english_date(
            self._region_value(ocr_result, "issue_date")
        )
        fallback_issue_date = self._normalize_english_date(
            self._find_first(
                r"Date\s*[:;]?\s*([A-Za-z]+\s+\d{1,2}[,.]?\s*\d{4})",
                single_line,
            )
        )
        issue_date = None
        if self._is_valid_english_date(region_issue_date):
            issue_date = region_issue_date
        elif self._is_valid_english_date(fallback_issue_date):
            issue_date = fallback_issue_date
        else:
            issue_date = self._normalize_partial_issue_date(
                region_issue_date,
                fallback_issue_date,
            )
        residency_country = normalize_country(
            "United States of America"
            if re.search(
                r"resident of the united states(?: of america)?",
                single_line,
                re.IGNORECASE,
            )
            or self._contains_any(
                single_line,
                [
                    "united states of america for purposes of u.s. taxation",
                    "united states for purposes of u.s. taxation",
                ],
            )
            else None
        )

        fields = {
            "taxpayer_name": taxpayer_name,
            "tin": tin,
            "tax_year": tax_year,
            "issue_date": issue_date,
            "residency_country": residency_country,
            "residency_country_code": "US" if residency_country else None,
        }
        quality_checks = {
            "has_certification_text": self._contains_any(
                single_line,
                [
                    "resident of the united states of america for purposes of u.s. taxation",
                    "certification",
                    "i certify that",
                ],
            ),
            "has_irs_heading": self._contains_any(
                single_line,
                [
                    "department of the treasury",
                    "internal revenue service",
                ],
            ),
        }
        return ExtractedDocument(
            document_type=self.document_type,
            source_path=str(source_path),
            template_id=ocr_result.template_id,
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=[],
        )

    def _clean_taxpayer_value(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"Taxpayer\s*:?", " ", value, flags=re.IGNORECASE)
        cleaned = re.sub(r"TIN\s*:?.*$", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9 .'-]", " ", cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        return cleaned or None

    def _select_taxpayer_name_candidate(
        self,
        region_value: str | None,
        fallback_value: str | None,
        *,
        region_taxpayer_raw: str | None,
    ) -> str | None:
        if fallback_value is None:
            return region_value
        if region_value is None:
            return fallback_value

        region_raw = (region_taxpayer_raw or "").lower()
        region_normalized = region_value.lower()
        if (
            "tin" in region_raw
            or region_normalized.startswith("payer ")
            or " tin " in f" {region_normalized} "
            or len(region_value) + 4 < len(fallback_value)
        ):
            return fallback_value
        return region_value

    def _normalize_partial_issue_date(self, *values: str | None) -> str | None:
        for value in values:
            if not value:
                continue
            cleaned = self._normalize_whitespace(value)
            cleaned = re.sub(r"^Date\s*[:;]?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip(" ,.;:")
            month_year_match = re.search(r"(\d{1,2})\s*,?\s*(\d{4})", cleaned)
            if month_year_match:
                month_or_day, year = month_year_match.groups()
                return f"{month_or_day}, {year}"
            if re.search(r"\b\d{4}\b", cleaned):
                return cleaned
        return None
