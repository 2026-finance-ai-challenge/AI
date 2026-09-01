from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

from k_market_ai.core.config import Settings
from k_market_ai.tax.service import TaxBundleDocument, TaxDocumentService

DOCUMENT_TYPES = (
    "RESIDENCY_CERTIFICATE",
    "APOSTILLE",
    "REDUCED_TAX_APPLICATION",
)


def media_type(path: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }[path.suffix.lower()]


async def run(paths: tuple[Path, Path, Path]) -> None:
    documents = tuple(
        TaxBundleDocument(
            document_type=document_type,
            file_name=path.name,
            content_type=media_type(path),
            document_base64=base64.b64encode(path.read_bytes()).decode("ascii"),
        )
        for document_type, path in zip(DOCUMENT_TYPES, paths, strict=True)
    )
    result = await TaxDocumentService(Settings()).verify_bundle(
        documents,
        expected_residency_country="US",
        investor_type="INDIVIDUAL",
        safety_identifier="0" * 64,
    )
    print(
        json.dumps(
            {
                "verification_status": result.verification_status,
                "cross_check": result.cross_check,
                "findings": [finding.model_dump(mode="json") for finding in result.findings],
                "documents": [
                    {
                        "document_type": document.detected_document_type,
                        "verification_status": document.verification_status,
                        "fields": document.fields.model_dump(mode="json"),
                        "ocr_confidence": document.ocr_confidence,
                        "issues": [issue.model_dump(mode="json") for issue in document.issues],
                    }
                    for document in result.documents
                ],
                "model": result.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="세무 문서 3종 OCR·교차검증 하네스")
    parser.add_argument("residency_certificate", type=Path)
    parser.add_argument("apostille", type=Path)
    parser.add_argument("reduced_tax_application", type=Path)
    arguments = parser.parse_args()
    asyncio.run(
        run(
            (
                arguments.residency_certificate,
                arguments.apostille,
                arguments.reduced_tax_application,
            )
        )
    )


if __name__ == "__main__":
    main()
