from __future__ import annotations

from k_market_ai.tax.document_model.parsers.apostille import ApostilleParser
from k_market_ai.tax.document_model.parsers.base import BaseDocumentParser
from k_market_ai.tax.document_model.parsers.residency_certificate import ResidencyCertificateParser
from k_market_ai.tax.document_model.parsers.withholding_tax_form import WithholdingTaxFormParser
from k_market_ai.tax.document_model.schemas import DocumentType


def build_parser_registry() -> dict[DocumentType, BaseDocumentParser]:
    return {
        DocumentType.RESIDENCY_CERTIFICATE: ResidencyCertificateParser(),
        DocumentType.APOSTILLE: ApostilleParser(),
        DocumentType.WITHHOLDING_TAX_FORM: WithholdingTaxFormParser(),
    }
