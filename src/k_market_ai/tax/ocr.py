from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from k_market_ai.core.errors import AppError

MAX_PDF_PAGES = 5
OCR_TIMEOUT_SECONDS = 45
PDF_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    confidence: float


class TaxOcrEngine:
    """로컬 OCR 바이너리만 사용해 문서를 읽고 원문은 즉시 폐기한다."""

    def __init__(self) -> None:
        self._tesseract = shutil.which("tesseract")
        self._pdftoppm = shutil.which("pdftoppm")

    def read(self, content: bytes, content_type: str, file_name: str) -> OcrResult:
        if self._tesseract is None:
            raise AppError(
                code="OCR_ENGINE_UNAVAILABLE",
                message="The document OCR runtime is unavailable.",
                status_code=503,
            )
        suffix = _validated_suffix(content, content_type, file_name)
        with tempfile.TemporaryDirectory(prefix="kmarket-tax-ocr-") as temporary:
            root = Path(temporary)
            source = root / f"source{suffix}"
            source.write_bytes(content)
            pages = self._pages(source, suffix, root)
            language = self._language()
            words: list[str] = []
            confidences: list[float] = []
            for page in pages:
                completed = _run(
                    [
                        self._tesseract,
                        str(page),
                        "stdout",
                        "-l",
                        language,
                        "--oem",
                        "1",
                        "--psm",
                        "6",
                        "tsv",
                    ],
                    OCR_TIMEOUT_SECONDS,
                )
                page_words, page_confidences = _parse_tsv(completed.stdout)
                words.extend(page_words)
                confidences.extend(page_confidences)
            text = "\n".join(words).strip()
            if not text:
                raise AppError(
                    code="OCR_TEXT_UNREADABLE",
                    message="No readable text was found in the document.",
                    status_code=422,
                )
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return OcrResult(text=text, confidence=round(confidence / 100, 4))

    def _pages(self, source: Path, suffix: str, root: Path) -> list[Path]:
        if suffix != ".pdf":
            return [source]
        if self._pdftoppm is None:
            raise AppError(
                code="OCR_ENGINE_UNAVAILABLE",
                message="The PDF OCR runtime is unavailable.",
                status_code=503,
            )
        output = root / "page"
        _run(
            [
                self._pdftoppm,
                "-f",
                "1",
                "-l",
                str(MAX_PDF_PAGES),
                "-r",
                "250",
                "-png",
                str(source),
                str(output),
            ],
            PDF_TIMEOUT_SECONDS,
        )
        pages = sorted(root.glob("page-*.png"))
        if not pages:
            raise AppError(
                code="OCR_INPUT_UNREADABLE",
                message="The PDF could not be rendered for OCR.",
                status_code=422,
            )
        return pages

    def _language(self) -> str:
        if self._tesseract is None:
            raise RuntimeError("OCR runtime was not initialized.")
        completed = _run([self._tesseract, "--list-langs"], 5)
        languages = set(completed.stdout.split())
        if "eng" not in languages:
            raise AppError(
                code="OCR_ENGINE_UNAVAILABLE",
                message="The English OCR language pack is unavailable.",
                status_code=503,
            )
        return "eng+kor" if "kor" in languages else "eng"


def _validated_suffix(content: bytes, content_type: str, file_name: str) -> str:
    declared = content_type.lower().split(";", maxsplit=1)[0].strip()
    suffix = Path(file_name).suffix.lower()
    signatures = {
        "application/pdf": (b"%PDF-", ".pdf"),
        "image/png": (b"\x89PNG\r\n\x1a\n", ".png"),
        "image/jpeg": (b"\xff\xd8\xff", ".jpg"),
    }
    expected = signatures.get(declared)
    if expected is None or not content.startswith(expected[0]):
        raise AppError(
            code="INVALID_DOCUMENT",
            message="The document signature does not match its content type.",
            status_code=400,
        )
    allowed_suffixes = {
        "application/pdf": {".pdf"},
        "image/png": {".png"},
        "image/jpeg": {".jpg", ".jpeg"},
    }
    if suffix not in allowed_suffixes[declared]:
        raise AppError(
            code="INVALID_DOCUMENT",
            message="The document filename does not match its content type.",
            status_code=400,
        )
    return expected[1]


def _run(arguments: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        # 실행 파일 경로와 인자는 서버가 고정한다.
        return subprocess.run(  # noqa: S603
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exception:
        raise AppError(
            code="OCR_TIMEOUT",
            message="Document OCR exceeded the processing time limit.",
            status_code=503,
        ) from exception
    except subprocess.CalledProcessError as exception:
        raise AppError(
            code="OCR_INPUT_UNREADABLE",
            message="The document could not be processed by OCR.",
            status_code=422,
        ) from exception


def _parse_tsv(payload: str) -> tuple[list[str], list[float]]:
    lines: list[str] = []
    line_words: list[str] = []
    current_line: tuple[str, str, str, str] | None = None
    confidences: list[float] = []
    for row in csv.DictReader(io.StringIO(payload), delimiter="\t"):
        value = (row.get("text") or "").strip()
        if not value:
            continue
        line_key = (
            row.get("page_num") or "",
            row.get("block_num") or "",
            row.get("par_num") or "",
            row.get("line_num") or "",
        )
        if current_line is not None and line_key != current_line:
            lines.append(" ".join(line_words))
            line_words = []
        current_line = line_key
        line_words.append(value)
        try:
            confidence = float(row.get("conf") or "-1")
        except ValueError:
            continue
        if confidence >= 0:
            confidences.append(confidence)
    if line_words:
        lines.append(" ".join(line_words))
    return lines, confidences


def field_after(text: str, labels: tuple[str, ...], maximum: int = 120) -> str | None:
    alternatives = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?:{alternatives})\s*(?::|：|-)?\s*([^\n]{{1,{maximum}}})",
        text,
        flags=re.IGNORECASE,
    )
    return _clean_field(match.group(1)) if match else None


def first_date(text: str) -> str | None:
    match = re.search(r"\b(20\d{2})[./-](0?[1-9]|1[0-2])[./-](0?[1-9]|[12]\d|3[01])\b", text)
    if match is None:
        return None
    return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def country_code(text: str, expected: str) -> str | None:
    names = {
        "US": ("UNITED STATES", "USA", "U.S.A", "미국"),
        "GB": ("UNITED KINGDOM", "UK", "GREAT BRITAIN", "영국"),
        "JP": ("JAPAN", "일본"),
        "CN": ("CHINA", "PEOPLE'S REPUBLIC OF CHINA", "중국"),
        "SG": ("SINGAPORE", "싱가포르"),
        "KR": ("REPUBLIC OF KOREA", "SOUTH KOREA", "대한민국"),
    }
    normalized = text.upper()
    candidates = names.get(expected.upper(), (expected.upper(),))
    if any(candidate.upper() in normalized for candidate in candidates):
        return expected.upper()
    return None


def _clean_field(value: str) -> str | None:
    cleaned = re.split(
        r"\s{2,}|\b(?:Date|Country|Authority|Number|Address|TIN|Tax year)\b\s*[:：]",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" :：-|,;")
    return cleaned or None
