from __future__ import annotations

import re
from pathlib import Path

from k_market_ai.tax.document_model.normalization import (
    canonicalize_name,
    normalize_address,
    normalize_country,
    normalize_country_code,
    normalize_percentage,
)
from k_market_ai.tax.document_model.parsers.base import BaseDocumentParser
from k_market_ai.tax.document_model.schemas import DocumentType, ExtractedDocument, OCRResult


class WithholdingTaxFormParser(BaseDocumentParser):
    document_type = DocumentType.WITHHOLDING_TAX_FORM
    _LABEL_FRAGMENTS = {"last", "ast", "first", "irst", "middle", "iddle", "name"}
    _PHONE_CONTEXT_PATTERN = re.compile(r"(?:\+?\d{1,2}-)$")
    _AMBIGUOUS_ALNUM_DIGIT_MAP = {
        "O": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        single_line = self._normalize_whitespace(text)
        region_address_raw = self._region_value(ocr_result, "address")
        region_address = self._sanitize_address_candidate(region_address_raw)
        full_text_address_raw = self._find_first(
            r"(\d{1,5}\s+[A-Za-z0-9 ,.'#-]+?(?:United States of America|USA))",
            single_line,
        ) or self._find_first(
            r"주소(?:\s*\([^)]+\))?\s*([0-9A-Za-z ,.'#-]+?(?:United States of America|USA))",
            single_line,
        )
        full_text_address = self._sanitize_address_candidate(full_text_address_raw)
        region_tin = self._region_value(ocr_result, "tin")
        dividend_rate_source = self._normalize_whitespace(
            " ".join(
                part
                for part in [self._region_value(ocr_result, "dividend_tax_rate"), single_line]
                if part
            )
        )
        applicant_name = self._normalize_name(
            self._clean_name_value(self._region_value(ocr_result, "applicant_name"))
            or self._find_first(
                r"신청인\s+([A-Za-z][A-Za-z .'-]+?)\s+\(?서명",
                single_line,
            )
            or self._find_first(r"신청인\s+([A-Za-z][A-Za-z .'-]+)", single_line)
        )
        derived_first_name, derived_middle_name, derived_last_name = self._derive_name_parts(
            applicant_name
        )

        last_name = self._normalize_name(
            self._compact_name_value(self._region_value(ocr_result, "last_name"))
            or derived_last_name
            or self._find_first(r"Last Name\)?\s*([A-Z]{2,})\b", single_line)
            or self._find_first(
                r"Last Name\)?\s*([A-Za-z][A-Za-z .'-]+?)\s+\(?First Name\)?",
                single_line,
            )
        )
        first_name = self._normalize_name(
            self._compact_name_value(self._region_value(ocr_result, "first_name"))
            or derived_first_name
            or self._find_first(r"First Name\)?\s*([A-Z]{2,})\b", single_line)
            or self._find_first(
                r"First Name\)?\s*([A-Za-z][A-Za-z .'-]+?)\s+\(?Middle Name\)?",
                single_line,
            )
        )
        middle_name = self._normalize_name(
            self._compact_name_value(self._region_value(ocr_result, "middle_name"))
            or derived_middle_name
            or self._find_first(r"Middle Name\)?\s*([A-Z])\b", single_line)
        )
        last_name = self._prefer_applicant_last_name(
            last_name,
            applicant_name=applicant_name,
        )
        tin = self._extract_best_tin(
            single_line,
            region_tin=region_tin,
        )
        address = normalize_address(
            self._merge_address_candidates(
                region_address=region_address,
                full_text_address=full_text_address,
                raw_full_text_address=full_text_address_raw,
            )
        )
        tin = tin or self._extract_tin_from_address_noise(
            region_address_raw,
            full_text_address_raw,
            address,
        )
        birth_date = self._extract_iso_date(
            self._region_value(ocr_result, "birth_date")
            or self._find_first(
                r"(?:생년월일|Date\s+of\s+Birth)\s*[:;]?\s*((?:19|20)\d{2}[-./]\d{1,2}[-./]\d{1,2})",
                single_line,
            )
        )
        phone_number = self._extract_phone_number(
            self._region_value(ocr_result, "phone_number")
            or self._find_first(
                r"(?:전화번호|거주지\s*전화|Phone|Tel)\s*[:;]?\s*(\+?\d[\d\s()-]{7,20}\d)",
                single_line,
            )
        )
        region_country = self._region_value(ocr_result, "residency_country")
        country_code = normalize_country_code(
            self._region_value(ocr_result, "residency_country_code")
        ) or normalize_country_code(
            self._find_first(r"거주지국코드\s*([A-Z]{2})", single_line)
            or self._find_first(r"Country Code\s*[:;]?\s*([A-Z]{2})", single_line)
            or region_country
            or address
            or single_line
        )
        country = self._select_country_candidate(
            region_country=region_country,
            full_text_country=single_line,
            address=address,
            country_code=country_code,
        )
        if country == "United States of America":
            country_code = "US"
        else:
            country_code = country_code or normalize_country_code(country)
        dividend_tax_rate = normalize_percentage(
            self._extract_first_pattern(
                [
                    r"배당소득\s*세율\s*(\d{1,2}\s*%)",
                    r"배당소득\s*세율\s*(\d{1,2})(?=\s*(?:양도소득|%))",
                    r"Dividend(?:\s+Income)?\s+Tax\s+Rate\s*(\d{1,2}\s*%)",
                    r"(15\s*%)",
                    r"\b(15)\b",
                ],
                dividend_rate_source,
            )
        )
        region_signature_date = self._normalize_signature_date_region(
            self._region_value(ocr_result, "signature_date")
        )
        fallback_signature_date = self._extract_signature_date_fallback(single_line)
        signature_date = self._select_signature_date_candidate(
            region_signature_date,
            fallback_signature_date,
        )
        middle_name = self._derive_middle_name(
            middle_name,
            applicant_name=applicant_name,
            first_name=first_name,
            last_name=last_name,
        )
        first_name = self._prefer_applicant_first_name(
            first_name,
            applicant_name=applicant_name,
            middle_name=middle_name,
            last_name=last_name,
        )
        applicant_name = self._rebuild_applicant_name(
            applicant_name,
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
        )

        fields = {
            "first_name": first_name,
            "last_name": last_name,
            "middle_name": middle_name,
            "tin": tin,
            "birth_date": birth_date,
            "phone_number": phone_number,
            "address": address,
            "residency_country": country,
            "residency_country_code": country_code,
            "dividend_tax_rate": dividend_tax_rate,
            "signature_date": signature_date,
            "applicant_name": applicant_name,
        }
        quality_checks = {
            "contains_non_resident_heading": self._contains_any(
                single_line,
                ["국내원천소득 제한세율 적용신청서", "비거주자용"],
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

    def _extract_iso_date(self, value: str | None) -> str | None:
        if not value:
            return None
        match = re.search(r"((?:19|20)\d{2})[-./]\s*(\d{1,2})[-./]\s*(\d{1,2})", value)
        if not match:
            return None
        return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"

    def _extract_phone_number(self, value: str | None) -> str | None:
        if not value:
            return None
        match = re.search(r"\+?\d[\d\s()-]{7,20}\d", value)
        if not match:
            return None
        normalized = re.sub(r"[^\d+]", "", match.group(0))
        return normalized if normalized.startswith("+") else normalized

    def _merge_address_candidates(
        self,
        *,
        region_address: str | None,
        full_text_address: str | None,
        raw_full_text_address: str | None,
    ) -> str | None:
        if region_address and full_text_address:
            sanitized_full_text_address = self._sanitize_full_text_address(full_text_address)
            full_text_street_number = self._leading_street_number(full_text_address)
            repaired_region_address = self._repair_address_spacing(region_address) or region_address
            if full_text_street_number and not self._leading_street_number(region_address):
                repaired_region_address = f"{full_text_street_number} {region_address}"
            if self._contains_leading_name_noise(raw_full_text_address):
                region_has_suite_number = re.search(
                    r"\bSuite\s+[0-9][A-Za-z0-9-]*",
                    region_address,
                    re.IGNORECASE,
                )
                sanitized_has_suite_number = sanitized_full_text_address and re.search(
                    r"\bSuite\s+[0-9][A-Za-z0-9-]*",
                    sanitized_full_text_address,
                    re.IGNORECASE,
                )
                if sanitized_full_text_address and (
                    not region_has_suite_number or sanitized_has_suite_number
                ):
                    return sanitized_full_text_address
                return repaired_region_address
            if (
                repaired_region_address
                and sanitized_full_text_address
                and self._has_address_street_marker(repaired_region_address)
                and not self._has_address_street_marker(sanitized_full_text_address)
            ):
                return repaired_region_address
            return sanitized_full_text_address or full_text_address
        return (
            self._sanitize_full_text_address(full_text_address)
            or self._repair_address_spacing(region_address)
            or region_address
        )

    def _leading_street_number(self, value: str | None) -> str | None:
        if not value:
            return None
        match = re.match(r"^(\d{1,5})\b", value)
        if not match:
            return None
        return match.group(1)

    def _contains_leading_name_noise(self, value: str | None) -> bool:
        if not value:
            return False
        return re.search(r"^\d{1,5}\s+USER\b", value, re.IGNORECASE) is not None

    def _sanitize_full_text_address(self, value: str | None) -> str | None:
        if not value:
            return None
        sanitized = re.sub(r"^\d{1,5}\s+USER\b\s+", "", value, flags=re.IGNORECASE)
        return self._sanitize_address_candidate(sanitized)

    def _sanitize_address_candidate(self, value: str | None) -> str | None:
        normalized = normalize_address(value)
        if not normalized:
            return None
        normalized = self._repair_address_spacing(normalized) or normalized
        normalized = self._strip_leading_address_noise(normalized)
        return normalized or None

    def _strip_leading_address_noise(self, value: str) -> str:
        label_matches = list(
            re.finditer(
                r"\b(?:USER|Last Name|First Name|Middle Name)\b",
                value,
                re.IGNORECASE,
            )
        )
        if label_matches:
            search_start = label_matches[-1].end()
            street_match = re.search(r"\b\d{1,5}\s+[A-Za-z]", value[search_start:])
            if street_match:
                return value[search_start + street_match.start() :]
            return value
        return value

    def _repair_address_spacing(self, value: str | None) -> str | None:
        if not value:
            return None
        repaired = re.sub(r"^(\d{1,5})([A-Za-z])", r"\1 \2", value)
        return repaired or None

    def _has_address_street_marker(self, value: str | None) -> bool:
        if not value:
            return False
        return (
            re.search(
                r"\b(street|st|road|rd|avenue|ave|blvd|boulevard|suite|apt)\b",
                value,
                re.IGNORECASE,
            )
            is not None
        )

    def _extract_tin_from_address_noise(self, *candidates: str | None) -> str | None:
        for candidate in candidates:
            if not candidate:
                continue
            tin = self._extract_tin_from_text(candidate)
            if tin:
                return tin
        return None

    def _extract_best_tin(
        self,
        full_text: str,
        *,
        region_tin: str | None,
    ) -> str | None:
        exact_full_text_tin = self._extract_tin_from_text(full_text)
        if exact_full_text_tin:
            return exact_full_text_tin

        exact_region_tin = self._extract_tin_from_text(region_tin)
        if exact_region_tin:
            return exact_region_tin

        partial_region_tin = self._extract_partial_tin_from_text(region_tin)
        compact_partial_region_tin = self._extract_compact_partial_tin_from_text(region_tin)
        partial_full_text_tin = self._extract_partial_tin_from_text(full_text)
        overlong_full_text_tin = self._extract_overlong_tin_from_text(full_text)

        if overlong_full_text_tin:
            trusted_partial_tin = partial_region_tin or compact_partial_region_tin
            if trusted_partial_tin and not overlong_full_text_tin.startswith(trusted_partial_tin):
                return trusted_partial_tin
            return overlong_full_text_tin

        return partial_region_tin or compact_partial_region_tin or partial_full_text_tin

    def _extract_tin_from_text(
        self,
        text: str | None,
        *,
        allow_overlong: bool = False,
    ) -> str | None:
        if not text:
            return None
        patterns = [
            r"\b(\d{3}-\d{2}-\d{4})\b",
            r"\b(\d{2}-\d{7})\b",
        ]
        if allow_overlong:
            patterns.extend(
                [
                    r"(\d{3}-\d{2}-\d{4})\d\b",
                    r"(\d{2}-\d{7})\d\b",
                ]
            )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if self._tin_match_has_phone_context(text, match.start()):
                    continue
                return next(
                    (group.strip() for group in match.groups() if group),
                    match.group(0).strip(),
                )
        return None

    def _extract_overlong_tin_from_text(self, text: str | None) -> str | None:
        if not text:
            return None
        patterns = [
            r"(\d{3}-\d{2}-\d{4})\d\b",
            r"(\d{2}-\d{7})\d\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if self._tin_match_has_phone_context(text, match.start()):
                    continue
                return next(
                    (group.strip() for group in match.groups() if group),
                    match.group(0).strip(),
                )
        return None

    def _extract_partial_tin_from_text(self, text: str | None) -> str | None:
        if not text:
            return None
        patterns = [
            r"\b(\d{3}-\d{2}-\d{3})\b",
            r"\b(\d{2}-\d{6})\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if self._tin_match_has_phone_context(text, match.start()):
                    continue
                return next(
                    (group.strip() for group in match.groups() if group),
                    match.group(0).strip(),
                )
        return None

    def _extract_compact_partial_tin_from_text(self, text: str | None) -> str | None:
        if not text:
            return None
        pattern = r"\b(\d{3})\s*-\s*(\d{2})\s*-?\s*(\d{2,3})\b"
        for match in re.finditer(pattern, text):
            if self._tin_match_has_phone_context(text, match.start()):
                continue
            if re.fullmatch(r"\d{3}-\d{4}", match.group(0).strip()):
                continue
            first, second, trailing = match.groups()
            return f"{first}-{second}-{trailing}"
        return None

    def _tin_match_has_phone_context(self, text: str, start: int) -> bool:
        prefix = text[max(0, start - 4) : start]
        if self._PHONE_CONTEXT_PATTERN.search(prefix):
            return True
        previous_context = text[max(0, start - 20) : start]
        return re.search(r"(전화|phone|tel)\s*$", previous_context, re.IGNORECASE) is not None

    def _select_signature_date_candidate(
        self,
        region_signature_date: str | None,
        fallback_signature_date: str | None,
    ) -> str | None:
        if not self._is_plausible_signature_date(region_signature_date):
            return fallback_signature_date
        if not self._is_plausible_signature_date(fallback_signature_date):
            return region_signature_date

        region_year = int(str(region_signature_date)[:4])
        if region_year < 1900 or region_year > 2100:
            return fallback_signature_date
        fallback_year = int(str(fallback_signature_date)[:4])
        if fallback_year > region_year:
            return fallback_signature_date
        return region_signature_date

    def _extract_signature_date_fallback(self, text: str) -> str | None:
        anchored_candidates = self._extract_valid_iso_date_candidates(
            r"((?:19|20)\d{2}\s*[-./년]?\s*\d{1,2}\s*[-./월]?\s*\d{1,2}\s*일?)"
            r"(?=[^\d]{0,24}(?:신청인|서명|또는\s*인|인\b))",
            text,
        )
        if anchored_candidates:
            return anchored_candidates[-1]

        prefilled_candidate = self._extract_prefilled_signature_date_candidate(text)
        if prefilled_candidate:
            return prefilled_candidate

        fallback_candidates: list[str] = []
        for pattern in (
            r"((?:19|20)\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일)",
            r"((?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2})",
            r"((?:19|20)\d{2}\s+\d{1,2}\s+\d{1,2})",
        ):
            fallback_candidates.extend(self._extract_valid_iso_date_candidates(pattern, text))
        if fallback_candidates:
            return fallback_candidates[-1]
        return self._normalize_iso_date(self._find_first(r"(\d{4}-\d{2}\d{2})", text))

    def _extract_prefilled_signature_date_candidate(self, text: str) -> str | None:
        field_header = re.search(
            r"(?:\bLast\s+Name\b|\bFirst\s+Name\b|\bMiddle\s+Name\b)",
            text,
            re.IGNORECASE,
        )
        if field_header is None:
            return None

        header_text = text[: field_header.start()]
        candidates: list[str] = []
        for pattern in (
            r"((?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2})",
            r"((?:19|20)\d{2}\s+\d{1,2}\s+\d{1,2})",
        ):
            candidates.extend(self._extract_valid_iso_date_candidates(pattern, header_text))
        return candidates[-1] if candidates else None

    def _extract_valid_iso_date_candidates(
        self,
        pattern: str,
        text: str,
    ) -> list[str]:
        candidates: list[str] = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = self._normalize_iso_date(match.group(1))
            if not self._is_valid_iso_date(value):
                continue
            if value in candidates:
                continue
            candidates.append(value)
        return candidates

    def _select_country_candidate(
        self,
        *,
        region_country: str | None,
        full_text_country: str | None,
        address: str | None,
        country_code: str | None,
    ) -> str | None:
        for candidate in [region_country, full_text_country, address]:
            normalized = normalize_country(candidate)
            if normalized == "United States of America":
                return normalized
        if country_code == "US":
            return "United States of America"
        for candidate in [region_country, address]:
            normalized = normalize_country(candidate)
            if normalized:
                return normalized
        return None

    def _clean_name_value(self, value: str | None) -> str | None:
        if not value:
            return None
        # 영역 경계에서 첫 글자가 잘린 영문 필드명은 값이 아닌 라벨로 제거한다.
        cleaned = re.sub(r"\b(L?ast|F?irst|M?iddle|Name)\b", " ", value, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9 .'-]", " ", cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        return cleaned or None

    def _compact_name_value(self, value: str | None) -> str | None:
        cleaned = self._clean_name_value(value)
        if not cleaned:
            return None
        tokens = cleaned.split()
        if not tokens:
            return None
        lowered_tokens = [token.strip(".").lower() for token in tokens]
        if any(token in self._LABEL_FRAGMENTS for token in lowered_tokens):
            return None
        if len(tokens) == 1:
            return tokens[0]
        if len(tokens) == 2 and all(len(token) <= 10 for token in tokens):
            return tokens[0]
        return None

    def _derive_name_parts(
        self,
        applicant_name: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        if not applicant_name:
            return None, None, None
        tokens = [token.strip(".") for token in applicant_name.split() if token.strip(".")]
        if len(tokens) < 2:
            return None, None, None
        first_name = self._normalize_alphanumeric_name_token(tokens[0])
        last_name = tokens[-1]
        middle_name = tokens[1][:1].upper() if len(tokens) >= 3 else None
        return first_name, middle_name, last_name

    def _derive_middle_name(
        self,
        middle_name: str | None,
        *,
        applicant_name: str | None,
        first_name: str | None,
        last_name: str | None,
    ) -> str | None:
        normalized_middle_name = self._normalize_middle_initial(middle_name)
        derived_initial: str | None = None
        if applicant_name and first_name and last_name:
            tokens = [token.strip(".") for token in applicant_name.split()]
            first_name_matches = bool(tokens) and (
                tokens[0].lower() == first_name.lower()
                or self._looks_like_truncated_name_token(first_name, tokens[0])
            )
            if len(tokens) >= 3 and tokens[-1].lower() == last_name.lower() and first_name_matches:
                derived_initial = self._normalize_middle_initial(tokens[1][:1].upper())
            elif len(tokens) == 2 and first_name_matches:
                compact_second_token = re.sub(r"[^A-Za-z]", "", tokens[1])
                compact_last_name = re.sub(r"[^A-Za-z]", "", last_name)
                if (
                    compact_second_token
                    and compact_last_name
                    and len(compact_second_token) > len(compact_last_name)
                    and compact_second_token.lower().endswith(compact_last_name.lower())
                ):
                    compact_prefix = compact_second_token[: -len(compact_last_name)]
                    if len(compact_prefix) == 1 and compact_prefix.isalpha():
                        derived_initial = self._normalize_middle_initial(compact_prefix.upper())
        if derived_initial and self._looks_like_middle_name_bleed(normalized_middle_name):
            return derived_initial
        if (
            normalized_middle_name
            and len(normalized_middle_name.split()) == 1
            and len(normalized_middle_name) <= 4
        ):
            if (
                derived_initial
                and len(normalized_middle_name) == 1
                and normalized_middle_name.isalpha()
                and normalized_middle_name != derived_initial
            ):
                return derived_initial
            return normalized_middle_name
        if not applicant_name or not first_name or not last_name:
            return normalized_middle_name
        tokens = [token.strip(".") for token in applicant_name.split()]
        if len(tokens) < 3:
            return normalized_middle_name
        if tokens[-1].lower() == last_name.lower() and (
            tokens[0].lower() == first_name.lower()
            or not normalized_middle_name
            or len(normalized_middle_name) != 1
        ):
            return self._normalize_middle_initial(tokens[1][:1].upper())
        return normalized_middle_name

    def _looks_like_middle_name_bleed(self, value: str | None) -> bool:
        if not value:
            return False
        lowered = value.strip().strip(".").lower()
        if len(lowered) <= 1:
            return False
        bleed_roots = ("america", "united", "states", "middle", "name")
        return any(root.startswith(lowered) or lowered in root for root in bleed_roots)

    def _rebuild_applicant_name(
        self,
        applicant_name: str | None,
        *,
        first_name: str | None,
        middle_name: str | None,
        last_name: str | None,
    ) -> str | None:
        rebuilt = self._format_applicant_name(first_name, middle_name, last_name)
        if not rebuilt:
            return applicant_name
        if not applicant_name:
            return rebuilt
        if canonicalize_name(applicant_name) != canonicalize_name(rebuilt):
            return rebuilt
        if applicant_name != rebuilt and re.fullmatch(r"[A-Za-z '.-]+", rebuilt):
            return rebuilt
        return applicant_name

    def _prefer_applicant_first_name(
        self,
        first_name: str | None,
        *,
        applicant_name: str | None,
        middle_name: str | None,
        last_name: str | None,
    ) -> str | None:
        derived_first_name, derived_middle_name, derived_last_name = self._derive_name_parts(
            applicant_name
        )
        normalized_first_name = self._normalize_alphanumeric_name_token(first_name)
        normalized_derived_first_name = self._normalize_alphanumeric_name_token(derived_first_name)
        if not derived_first_name or not first_name:
            return normalized_first_name or normalized_derived_first_name
        if not last_name or derived_last_name != last_name:
            return normalized_first_name
        if middle_name and derived_middle_name and middle_name != derived_middle_name:
            if self._looks_like_truncated_name_token(
                normalized_first_name,
                normalized_derived_first_name,
            ):
                return normalized_derived_first_name
            return normalized_first_name
        if canonicalize_name(normalized_derived_first_name) == canonicalize_name(
            normalized_first_name
        ):
            return normalized_first_name
        if not re.search(r"[A-Za-z]", normalized_derived_first_name or ""):
            return normalized_first_name
        if self._looks_like_truncated_name_token(
            normalized_first_name,
            normalized_derived_first_name,
        ):
            return normalized_derived_first_name
        if self._digit_count(normalized_derived_first_name) > self._digit_count(
            normalized_first_name
        ):
            return normalized_derived_first_name
        return normalized_first_name

    def _prefer_applicant_last_name(
        self,
        last_name: str | None,
        *,
        applicant_name: str | None,
    ) -> str | None:
        _, _, derived_last_name = self._derive_name_parts(applicant_name)
        normalized_last_name = self._normalize_name(last_name)
        if not derived_last_name:
            return normalized_last_name
        if not normalized_last_name:
            return derived_last_name
        if canonicalize_name(normalized_last_name) == canonicalize_name(derived_last_name):
            return normalized_last_name
        if self._looks_like_label_bleed_last_name(
            normalized_last_name
        ) or self._looks_like_address_bleed_last_name(normalized_last_name):
            return derived_last_name
        return normalized_last_name

    def _looks_like_label_bleed_last_name(self, value: str | None) -> bool:
        if not value:
            return False
        normalized = canonicalize_name(value)
        if not normalized:
            return False
        if normalized in {canonicalize_name(fragment) for fragment in self._LABEL_FRAGMENTS}:
            return True
        suspicious_suffixes = ("ast", "irst", "iddle", "name")
        return any(
            normalized.endswith(fragment) and len(normalized) <= len(fragment) + 1
            for fragment in suspicious_suffixes
        )

    def _looks_like_address_bleed_last_name(self, value: str | None) -> bool:
        normalized = canonicalize_name(value)
        if not normalized:
            return False
        return normalized in {"st", "rd", "ave", "blvd", "america", "united", "states"}

    def _format_applicant_name(
        self,
        first_name: str | None,
        middle_name: str | None,
        last_name: str | None,
    ) -> str | None:
        if not first_name and not middle_name and not last_name:
            return None
        formatted_middle = middle_name
        if (
            middle_name
            and len(middle_name) == 1
            and middle_name.isalpha()
            and (first_name or "").isalpha()
            and (last_name or "").isalpha()
        ):
            formatted_middle = f"{middle_name}."
        return " ".join(part for part in [first_name, formatted_middle, last_name] if part).strip()

    def _normalize_middle_initial(self, value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.strip().strip(".")
        if len(normalized) != 1:
            return value
        if normalized == "0":
            return "O"
        if normalized == "1":
            return "I"
        return normalized.upper()

    def _normalize_alphanumeric_name_token(self, value: str | None) -> str | None:
        if not value:
            return None
        token = value.strip().strip(".").upper()
        if not re.fullmatch(r"[A-Z0-9]+", token):
            return value
        first_digit_index = next((index for index, char in enumerate(token) if char.isdigit()), -1)
        if first_digit_index <= 0:
            return value
        prefix = token[:first_digit_index]
        suffix = token[first_digit_index:]
        normalized_suffix = "".join(
            self._AMBIGUOUS_ALNUM_DIGIT_MAP.get(char, char) for char in suffix
        )
        if normalized_suffix == suffix:
            return value
        if not normalized_suffix.isdigit():
            return value
        normalized = prefix + normalized_suffix
        if self._digit_count(normalized) <= self._digit_count(token):
            return value
        return normalized

    def _looks_like_truncated_name_token(
        self,
        candidate: str | None,
        reference: str | None,
    ) -> bool:
        if not candidate or not reference:
            return False
        normalized_candidate = canonicalize_name(candidate)
        normalized_reference = canonicalize_name(reference)
        if not normalized_candidate or not normalized_reference:
            return False
        return len(normalized_reference) > len(
            normalized_candidate
        ) and normalized_reference.endswith(normalized_candidate)

    def _normalize_signature_date_region(self, value: str | None) -> str | None:
        normalized = self._normalize_iso_date(value)
        if self._is_plausible_signature_date(normalized):
            return normalized
        if not value:
            return normalized
        year_first_candidate = self._extract_noisy_year_first_date(value)
        if self._is_plausible_signature_date(year_first_candidate):
            return year_first_candidate
        split_candidate = self._extract_noisy_split_signature_date(value)
        if self._is_plausible_signature_date(split_candidate):
            return split_candidate
        numeric_tokens = re.findall(r"\d{1,4}", value)
        if len(numeric_tokens) < 3:
            return normalized
        for index in range(len(numeric_tokens) - 2):
            day, month, year = numeric_tokens[index : index + 3]
            if len(year) != 4:
                continue
            try:
                day_int = int(day)
                month_int = int(month)
                year_int = int(year)
            except ValueError:
                continue
            if not (1 <= day_int <= 31 and 1 <= month_int <= 12 and 1900 <= year_int <= 2100):
                continue
            candidate = f"{year_int:04d}-{month_int:02d}-{day_int:02d}"
            if self._is_plausible_signature_date(candidate):
                return candidate
        return normalized

    def _extract_noisy_split_signature_date(self, value: str | None) -> str | None:
        if not value:
            return None
        tokens = re.findall(r"[A-Za-z0-9]{1,4}", value)
        normalized_tokens = [
            "".join(self._AMBIGUOUS_ALNUM_DIGIT_MAP.get(char, char) for char in token.upper())
            for token in tokens
        ]
        for index, token in enumerate(normalized_tokens):
            if not re.fullmatch(r"(?:19|20)\d{2}", token):
                continue
            if index < 2:
                continue
            day_token = normalized_tokens[index - 2]
            month_token = normalized_tokens[index - 1]
            if not day_token.isdigit() or not month_token.isdigit():
                continue
            day = int(day_token)
            month = int(month_token)
            year = int(token)
            if 1 <= day <= 31 and 1 <= month <= 12:
                return f"{year:04d}-{month:02d}-{day:02d}"
        return None

    def _extract_noisy_year_first_date(self, value: str) -> str | None:
        pattern = (
            r"(?<!\d)((?:19|20)\d{2})(?!\d)"
            r"(?:[^\d]{1,12})"
            r"(?<![A-Za-z0-9])(\d{1,2})(?![A-Za-z0-9])"
            r"(?:[^\d]{1,12})"
            r"(?<![A-Za-z0-9])(\d{1,2})(?![A-Za-z0-9])"
        )
        for match in re.finditer(pattern, value):
            year, month, day = match.groups()
            candidate = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            if self._is_plausible_signature_date(candidate):
                return candidate
        return None

    def _is_plausible_signature_date(self, value: str | None) -> bool:
        if not self._is_valid_iso_date(value):
            return False
        if not value:
            return False
        try:
            year, month, day = (int(part) for part in value.split("-"))
        except ValueError:
            return False
        return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31

    def _digit_count(self, value: str | None) -> int:
        if not value:
            return 0
        return sum(character.isdigit() for character in value)
