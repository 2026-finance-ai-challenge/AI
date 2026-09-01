from .apostille import ApostilleParser
from .registry import build_parser_registry
from .residency_certificate import ResidencyCertificateParser
from .withholding_tax_form import WithholdingTaxFormParser

__all__ = [
    "ApostilleParser",
    "ResidencyCertificateParser",
    "WithholdingTaxFormParser",
    "build_parser_registry",
]
