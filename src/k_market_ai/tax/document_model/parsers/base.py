from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from k_market_ai.tax.document_model.normalization import (
    normalize_english_date,
    normalize_iso_date,
    normalize_name,
    normalize_whitespace,
)
from k_market_ai.tax.document_model.schemas import (
    DocumentType,
    ExtractedDocument,
    OCRPage,
    OCRResult,
    OCRWordBox,
)


class BaseDocumentParser(ABC):
    document_type: DocumentType

    @abstractmethod
    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        raise NotImplementedError

    @staticmethod
    def _find_first(pattern: str, text: str, *, flags: int = re.IGNORECASE) -> str | None:
        match = re.search(pattern, text, flags)
        if not match:
            return None
        return next((group.strip() for group in match.groups() if group), match.group(0).strip())

    @staticmethod
    def _contains_any(text: str, needles: list[str]) -> bool:
        lowered = text.lower()
        return any(needle.lower() in lowered for needle in needles)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return normalize_whitespace(text)

    @staticmethod
    def _normalize_name(value: str | None) -> str | None:
        return normalize_name(value)

    @staticmethod
    def _extract_first_pattern(
        patterns: list[str],
        text: str,
        *,
        flags: int = re.IGNORECASE,
    ) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags)
            if match:
                return next(
                    (group.strip() for group in match.groups() if group),
                    match.group(0).strip(),
                )
        return None

    @staticmethod
    def _normalize_english_date(value: str | None) -> str | None:
        return normalize_english_date(value)

    @staticmethod
    def _normalize_iso_date(value: str | None) -> str | None:
        return normalize_iso_date(value)

    @staticmethod
    def _is_valid_english_date(value: str | None) -> bool:
        if not value:
            return False
        return (
            re.search(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
                value,
                re.IGNORECASE,
            )
            is not None
            and re.search(r"\b\d{4}\b", value) is not None
        )

    @staticmethod
    def _is_valid_iso_date(value: str | None) -> bool:
        if not value:
            return False
        return re.search(r"\b\d{4}-\d{2}-\d{2}\b", value) is not None

    @staticmethod
    def _region_value(ocr_result: OCRResult, region_name: str) -> str | None:
        return ocr_result.region_text(region_name)

    @staticmethod
    def _region_text(
        ocr_result: OCRResult,
        box: tuple[float, float, float, float],
    ) -> str | None:
        if not ocr_result.pages:
            return None
        page = ocr_result.pages[0]
        page_size = BaseDocumentParser._page_size(page)
        if page_size is None:
            return None

        width, height = page_size
        positioned_words: list[tuple[float, float, str]] = []
        left, top, right, bottom = box
        for word in page.words:
            center = BaseDocumentParser._word_center(word)
            if center is None:
                continue
            center_x, center_y = center
            normalized_x = center_x / width
            normalized_y = center_y / height
            if left <= normalized_x <= right and top <= normalized_y <= bottom:
                positioned_words.append((normalized_y, normalized_x, word.text.strip()))

        if not positioned_words:
            return None

        positioned_words.sort(key=lambda item: (round(item[0], 3), item[1]))
        text = " ".join(text for _, _, text in positioned_words if text)
        return BaseDocumentParser._normalize_whitespace(text) or None

    @staticmethod
    def _page_size(page: OCRPage) -> tuple[float, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        for word in page.words:
            for point in word.points:
                if len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
        if not xs or not ys:
            return None
        return max(xs), max(ys)

    @staticmethod
    def _word_center(word: OCRWordBox) -> tuple[float, float] | None:
        if not word.points:
            return None
        x_values = [float(point[0]) for point in word.points if len(point) >= 2]
        y_values = [float(point[1]) for point in word.points if len(point) >= 2]
        if not x_values or not y_values:
            return None
        return sum(x_values) / len(x_values), sum(y_values) / len(y_values)
