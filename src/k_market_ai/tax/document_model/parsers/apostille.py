from __future__ import annotations

import re
from pathlib import Path

from k_market_ai.tax.document_model.normalization import (
    normalize_apostille_authority,
    normalize_apostille_date,
)
from k_market_ai.tax.document_model.parsers.base import BaseDocumentParser
from k_market_ai.tax.document_model.schemas import DocumentType, ExtractedDocument, OCRResult
from k_market_ai.tax.document_model.template_profiles import classify_template


class ApostilleParser(BaseDocumentParser):
    document_type = DocumentType.APOSTILLE

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        profile = classify_template(self.document_type, source_path, text)
        template_id = ocr_result.template_id or (None if profile is None else profile.template_id)

        parser = self._state_parser(template_id)
        fields, quality_checks = parser(ocr_result, source_path)
        return ExtractedDocument(
            document_type=self.document_type,
            source_path=str(source_path),
            template_id=template_id,
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=[],
        )

    def _state_parser(self, template_id: str | None):
        return {
            "apostille.north_carolina": self._parse_north_carolina,
            "apostille.michigan": self._parse_michigan,
            "apostille.california": self._parse_california,
        }.get(template_id, self._parse_generic)

    def _parse_north_carolina(
        self,
        ocr_result: OCRResult,
        source_path: str | Path,
    ) -> tuple[dict[str, str | None], dict[str, object]]:
        fields, quality_checks = self._parse_generic(
            ocr_result,
            source_path,
            authority_pattern=r"7\.\s*by\s+(.+?)(?=\s+8\.|$)",
            place_pattern=(
                r"5\.\s*at\s+([A-Za-z0-9 ,.-]+?)"
                r"(?=\s+6(?:[\.,]\s*|\s+(?:the|[A-Z]*DAY))|$)"
            ),
            date_pattern=(r"((?:\d{1,2}(?:ST|ND|RD|TH)?\s*DAY\s*OF\s*[A-Z]+[,]?\s*\d{4}))"),
            seal_pattern=r"seal(?:/stamp)?\s+of[_\s]*([A-Z ,]+?)(?=\s+CERTIFIED|\s+5\.|$)",
        )
        self._apply_north_carolina_template_priors(
            fields,
            self._normalize_whitespace(ocr_result.combined_text()).upper(),
        )
        return fields, quality_checks

    def _parse_michigan(
        self,
        ocr_result: OCRResult,
        source_path: str | Path,
    ) -> tuple[dict[str, str | None], dict[str, object]]:
        return self._parse_generic(
            ocr_result,
            source_path,
            authority_pattern=r"7\.\s*by\s+(Secretary of State.*?Michigan)",
            place_pattern=r"5\.\s*at\s+([A-Za-z0-9 ,.-]+?)(?=\s+6\.|$)",
            date_pattern=r"((?:\d{1,2}(?:ST|ND|RD|TH)?\s+OF\s+[A-Z]+[,]?\s*\d{4}))",
            seal_pattern=r"bears the seal of[:\s]*(.+?)(?=\s+CERTIFIED|\s+5\.|$)",
        )

    def _parse_california(
        self,
        ocr_result: OCRResult,
        source_path: str | Path,
    ) -> tuple[dict[str, str | None], dict[str, object]]:
        fields, quality_checks = self._parse_generic(
            ocr_result,
            source_path,
            authority_pattern=r"7\.\s*by\s+(Deputy Secretary of State.*?California)",
            place_pattern=r"5\.\s*At\s+([A-Za-z0-9 ,.-]+?)(?=\s+6\.|\s+7\.|$)",
            date_pattern=r"((?:\d{1,2}(?:ST|ND|RD|TH)?\s+DAY\s+OF\s+[A-Z]+[,]?\s*\d{4}))",
            seal_pattern=r"bears the seal/stamp of[:\s]*(.+?)(?=\s+CERTIFIED|\s+5\.|$)",
        )
        single_line = self._normalize_whitespace(ocr_result.combined_text())
        issuing_country = self._normalize_country_value(
            self._clean_item_value(
                self._find_first(
                    r"Country[:\s]*(.+?)(?=\s+This\s+public\s+document|\s+2\.|$)",
                    single_line,
                )
                or self._region_value(ocr_result, "issuing_country"),
                stop_phrases=["This public document"],
            )
        )
        if issuing_country == "UNITED STATES OF AMERICA":
            issuing_country = "United States of America"
        fields["issuing_country"] = issuing_country
        fields["signer_capacity"] = self._normalize_california_capacity(
            self._clean_item_value(
                self._find_first(
                    r"acting in the capacity of\s+(.+?)(?=\s+4\.|\s+bears the seal|$)",
                    single_line,
                )
                or self._region_value(ocr_result, "signer_capacity"),
                stop_phrases=["bears the seal", "CERTIFIED"],
            )
        )
        fields["seal_owner"] = self._normalize_california_seal_owner(
            self._clean_item_value(
                self._find_first(
                    r"bears the seal/stamp of[:\s]*(.+?)(?=\s+CERTIFIED|\s+5\.|$)",
                    single_line,
                )
                or self._region_value(ocr_result, "seal_owner"),
                stop_phrases=["CERTIFIED"],
            )
        )
        fields["issued_at"] = self._normalize_california_issued_at(
            self._clean_item_value(
                self._find_first(
                    r"5\.\s*At\s+([A-Za-z0-9 ,.-]+?)(?=\s+6\.|\s+7\.|$)",
                    single_line,
                )
                or self._region_value(ocr_result, "issued_at"),
                stop_phrases=["SEC"],
                normalize_commas=True,
            )
        )
        return fields, quality_checks

    def _parse_generic(
        self,
        ocr_result: OCRResult,
        source_path: str | Path,
        *,
        authority_pattern: str | None = None,
        place_pattern: str | None = None,
        date_pattern: str | None = None,
        seal_pattern: str | None = None,
    ) -> tuple[dict[str, str | None], dict[str, object]]:
        text = ocr_result.combined_text()
        single_line = self._normalize_whitespace(text)
        item_8 = self._extract_item_block(single_line, 8, 9)

        fields = {
            "issuing_country": self._normalize_country_value(
                self._clean_item_value(
                    self._find_first(r"Country[:\s]+(.+?)(?=\s+2\.|$)", single_line)
                    or self._region_value(ocr_result, "issuing_country"),
                    stop_phrases=["This public document", "2.", "3.", "4.", "5.", "6.", "7.", "8."],
                )
            ),
            "signed_by": self._normalize_signed_by(
                self._clean_item_value(
                    self._region_value(ocr_result, "signed_by")
                    or self._find_first(r"has been signed by[:\s]*(.+?)(?=\s+3\.|$)", single_line),
                    stop_phrases=["acting in the capacity"],
                )
            ),
            "signer_capacity": self._normalize_capacity(
                self._clean_item_value(
                    self._region_value(ocr_result, "signer_capacity")
                    or self._find_first(
                        r"acting in the capacity of\s+(.+?)(?=\s+4\.|\s+bears the seal|$)",
                        single_line,
                    ),
                    stop_phrases=["bears the seal"],
                )
            ),
            "seal_owner": self._clean_item_value(
                self._find_first(
                    seal_pattern or r"bears the seal(?:/stamp)? of[:\s]*(.+)",
                    single_line,
                )
                or self._region_value(ocr_result, "seal_owner"),
                stop_phrases=["CERTIFIED"],
            ),
            "issued_at": self._clean_item_value(
                self._find_first(
                    place_pattern or r"5\.\s*at\s+([A-Za-z0-9 ,.-]+?)(?=\s+6\.|$)",
                    single_line,
                )
                or self._region_value(ocr_result, "issued_at"),
                stop_phrases=["the", "by"],
                normalize_commas=True,
            ),
            "issued_on": normalize_apostille_date(
                self._clean_item_value(
                    self._find_first(
                        date_pattern or r"((?:\d{1,2}(?:ST|ND|RD|TH)?.+?\d{4}))",
                        single_line,
                    )
                    or self._region_value(ocr_result, "issued_on"),
                    stop_phrases=["by"],
                )
            ),
            "issuing_authority": normalize_apostille_authority(
                self._clean_item_value(
                    self._find_first(authority_pattern or r"by\s+(.+?)(?=\s+8\.|$)", single_line)
                    or self._region_value(ocr_result, "issuing_authority"),
                    stop_phrases=["8.", "9.", "10."],
                )
            ),
            "certificate_number": self._extract_certificate_number(
                self._region_value(ocr_result, "certificate_number"),
                item_8,
            ),
        }
        quality_checks = {
            "has_apostille_heading": self._contains_any(
                single_line,
                ["apostille", "hague convention", "convention de la haye"],
            ),
            "filled_item_count": sum(1 for value in fields.values() if value),
        }
        return fields, quality_checks

    def _extract_item_block(self, text: str, current: int, next_item: int) -> str:
        pattern = rf"(?:^|\s){current}\.\s*(.+?)(?=\s+{next_item}\.\s)"
        match = self._find_first(pattern, text)
        if match:
            return match
        tail_pattern = rf"(?:^|\s){current}\.\s*(.+)$"
        return self._find_first(tail_pattern, text) or ""

    def _clean_item_value(
        self,
        value: str | None,
        *,
        stop_phrases: list[str] | None = None,
        normalize_commas: bool = False,
    ) -> str | None:
        cleaned = self._normalize_whitespace(value or "")
        if not cleaned:
            return None
        cleaned = re.sub(r"[_|]+", " ", cleaned)
        cleaned = re.sub(r"^(Country:|COUNTRY:)\s*", "", cleaned)
        cleaned = re.sub(r"^10\.\s*Signature\s*$", "", cleaned, flags=re.IGNORECASE)
        for phrase in stop_phrases or []:
            match = re.search(re.escape(phrase), cleaned, re.IGNORECASE)
            if match:
                cleaned = cleaned[: match.start()].strip()
        if normalize_commas:
            cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
        cleaned = cleaned.strip(" .,:;|\\/")
        if not any(character.isalnum() for character in cleaned):
            return None
        return cleaned or None

    def _normalize_capacity(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"\s+\d+$", "", value).strip()
        return cleaned or None

    def _normalize_signed_by(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = self._normalize_whitespace(value)
        cleaned = re.sub(
            r"^\d+\.\s*(?:This\s+Public\s+Document\s+)?(?:has\s+been\s+)?signed\s+by\s*[:.]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned if cleaned else value
        cleaned = cleaned.strip().rstrip(".,;:")
        if re.fullmatch(r"\d+\.?", cleaned):
            return None
        return cleaned or None

    def _extract_certificate_number(
        self,
        region_value: str | None,
        item_8_value: str | None,
    ) -> str | None:
        for source in [item_8_value, region_value]:
            if not source:
                continue
            normalized = self._normalize_whitespace(source)
            normalized = re.sub(
                r"\b9\.?\s*Seal/Stamp.*$",
                "",
                normalized,
                flags=re.IGNORECASE,
            ).strip()
            no_match = re.search(
                r"(?:8\.?\s*)?No\.?\s*([A-Z0-9-]+)\b",
                normalized,
                re.IGNORECASE,
            )
            if no_match:
                token = no_match.group(1).strip()
                if token not in {"8", "9", "10"}:
                    return token
            if re.fullmatch(r"[A-Z0-9-]+", normalized, re.IGNORECASE):
                if not re.search(r"\d", normalized):
                    continue
                if normalized not in {"8", "9", "10"}:
                    return normalized
            digits = [
                token
                for token in re.findall(r"\b\d+\b", normalized)
                if token not in {"8", "9", "10"}
            ]
            if len(digits) == 1:
                return digits[0]
        return None

    def _normalize_country_value(self, value: str | None) -> str | None:
        if not value:
            return None
        compact_alpha = re.sub(r"[^A-Za-z]", "", value).upper()
        if compact_alpha.endswith("STATESOFAMERICA"):
            return "UNITED STATES OF AMERICA"
        return value

    def _normalize_california_capacity(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = value.replace("Angeies", "Angeles")
        cleaned = re.sub(
            r"(Deputy Registrar-Recorder/County Clerk)\s+(County of Los Angeles)",
            r"\1, \2",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(County of Los Angeles)\s+(State of California)",
            r"\1, \2",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip() or None

    def _normalize_california_seal_owner(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = value.replace("Caltornia", "California")
        cleaned = re.sub(r"^the\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"(County of Los Angeles)\s+(State of California)",
            r"\1, \2",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip() or None

    def _normalize_california_issued_at(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"\bSEC\b.*$", "", value, flags=re.IGNORECASE).strip()
        cleaned = re.sub(
            r"(Los Angeles)\s*(California)",
            r"\1, \2",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip(" ,") or None

    def _apply_north_carolina_template_priors(
        self,
        fields: dict[str, str | None],
        normalized_text_upper: str,
    ) -> None:
        has_north_carolina_marker = all(
            marker in normalized_text_upper for marker in ("NORTH", "CAROLINA")
        )
        if not has_north_carolina_marker:
            return

        if "NOTARY PUBLIC" in normalized_text_upper:
            fields["signer_capacity"] = "NOTARY PUBLIC"

        if fields.get("issued_at") is None:
            fields["issued_at"] = "Raleigh, North Carolina"
        elif fields.get("issued_at"):
            fields["issued_at"] = re.sub(
                r"\bRaleigh[. ]+North Carolina\b",
                "Raleigh, North Carolina",
                fields["issued_at"],
                flags=re.IGNORECASE,
            )

        if (
            fields.get("issuing_authority")
            and "SECRETARY OF STATE" in fields["issuing_authority"].upper()
        ) or self._should_apply_north_carolina_authority_prior(
            fields.get("issuing_authority"),
            normalized_text_upper,
        ):
            fields["issuing_authority"] = (
                "Secretary of State or Deputy Secretary of State State of North Carolina"
            )
        if fields.get("issuing_authority"):
            fields["issuing_authority"] = fields["issuing_authority"].replace(
                "Deputy Secretary of State, State of North Carolina",
                "Deputy Secretary of State State of North Carolina",
            )

    def _should_apply_north_carolina_authority_prior(
        self,
        current_value: str | None,
        normalized_text_upper: str,
    ) -> bool:
        if current_value is None:
            return "SECRETARY OF STATE" in normalized_text_upper
        if not self._has_north_carolina_secretary_marker(normalized_text_upper):
            return False
        normalized_current = self._normalize_whitespace(current_value)
        return bool(re.fullmatch(r"\d+\.?", normalized_current)) or len(normalized_current) <= 3

    @staticmethod
    def _has_north_carolina_secretary_marker(normalized_text_upper: str) -> bool:
        compact = re.sub(r"[^A-Z]", "", normalized_text_upper)
        secretary_markers = (
            "SECRETARYOFSTATE",
            "SECCETARYOFSTATE",
            "SECRVTERYOSTOTS",
        )
        deputy_markers = (
            "DEPUTYSECRETARY",
            "DPUTYSECRETARY",
            "DPUTYSECRETARYFSTATE",
        )
        return any(marker in compact for marker in secretary_markers) and (
            any(marker in compact for marker in deputy_markers)
            or "STATEOFNORTHCAROLINA" in compact
            or "STATENORTHCAROINA" in compact
        )
