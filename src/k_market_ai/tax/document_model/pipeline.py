from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from k_market_ai.tax.document_model.document_checks import compute_document_checks
from k_market_ai.tax.document_model.ocr import TesseractOCREngine
from k_market_ai.tax.document_model.parsers import build_parser_registry
from k_market_ai.tax.document_model.quality import average_ocr_confidence, compute_quality_metrics
from k_market_ai.tax.document_model.schemas import DocumentType, ExtractedDocument, OCRResult
from k_market_ai.tax.document_model.template_profiles import classify_template


class TaxDocumentPipelineResult(BaseModel):
    ocr_result: OCRResult
    extracted_document: ExtractedDocument
    ocr_confidence: float


class TaxDocumentPipeline:
    """템플릿 분류부터 품질 검사까지 실제 API와 하네스가 공유하는 파이프라인."""

    def __init__(
        self,
        *,
        ocr_engine: TesseractOCREngine | None = None,
        blur_threshold: float = 100.0,
        min_ocr_confidence: float = 0.75,
    ) -> None:
        self._ocr_engine = ocr_engine
        self._blur_threshold = blur_threshold
        self._min_ocr_confidence = min_ocr_confidence
        self._parsers = build_parser_registry()

    def process(
        self,
        document_type: DocumentType,
        source_path: str | Path,
        *,
        source_name: str | Path | None = None,
        ocr_result: OCRResult | None = None,
    ) -> TaxDocumentPipelineResult:
        path = Path(source_path)
        if ocr_result is None:
            if self._ocr_engine is None:
                raise RuntimeError("No OCR result was provided and no OCR engine is configured.")
            result = self._ocr_engine.run(path)
        else:
            result = ocr_result
        template_source = source_name or path
        profile = classify_template(
            document_type,
            template_source,
            result.combined_text(),
        )
        if profile is not None:
            result.template_id = result.template_id or profile.template_id
            if not result.regions and self._ocr_engine is not None:
                result.regions = self._ocr_engine.run_regions(path, profile.ocr_regions)

        extracted = self._parsers[document_type].parse(result, template_source)
        extracted.template_id = result.template_id
        extracted.quality_checks.update(
            compute_document_checks(
                document_type,
                path,
                template_id=result.template_id,
                ocr_text=result.combined_text(),
            )
        )
        extracted.quality_checks["detected_template_id"] = result.template_id
        quality_metrics = compute_quality_metrics(
            path,
            blur_threshold=self._blur_threshold,
        )
        confidence = average_ocr_confidence(result.pages)
        if confidence is not None:
            quality_metrics["average_ocr_confidence"] = confidence
            quality_metrics["low_ocr_confidence"] = confidence < self._min_ocr_confidence
        extracted.quality_checks.update(quality_metrics)

        return TaxDocumentPipelineResult(
            ocr_result=result,
            extracted_document=extracted,
            ocr_confidence=round(confidence if confidence is not None else 0.0, 4),
        )
