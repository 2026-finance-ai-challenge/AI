from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from k_market_ai.tax.document_model.schemas import DocumentType


@dataclass(frozen=True)
class RegionRule:
    left: float
    top: float
    right: float
    bottom: float
    min_dark_ratio: float
    dark_threshold: int = 200


@dataclass(frozen=True)
class OCRRegionSpec:
    name: str
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class DocumentTemplateProfile:
    template_id: str
    document_type: DocumentType
    filename_markers: tuple[str, ...] = ()
    text_markers: tuple[str, ...] = ()
    quality_regions: dict[str, RegionRule] = field(default_factory=dict)
    checkbox_regions: dict[str, tuple[RegionRule, ...]] = field(default_factory=dict)
    ocr_regions: tuple[OCRRegionSpec, ...] = ()


RESIDENCY_STANDARD = DocumentTemplateProfile(
    template_id="residency.irs_standard",
    document_type=DocumentType.RESIDENCY_CERTIFICATE,
    filename_markers=("거주자증명서", "treasury", "irs"),
    text_markers=("internal revenue service", "department of the treasury"),
    quality_regions={
        "seal_present": RegionRule(0.02, 0.02, 0.17, 0.17, min_dark_ratio=0.06),
        "signature_present": RegionRule(0.51, 0.80, 0.71, 0.88, min_dark_ratio=0.02),
    },
    ocr_regions=(
        OCRRegionSpec("taxpayer_name", 0.10, 0.25, 0.34, 0.31),
        OCRRegionSpec("tin", 0.10, 0.27, 0.28, 0.33),
        OCRRegionSpec("tax_year", 0.10, 0.30, 0.23, 0.36),
        OCRRegionSpec("issue_date", 0.60, 0.22, 0.90, 0.28),
    ),
)


WITHHOLDING_STANDARD = DocumentTemplateProfile(
    template_id="withholding.non_resident_v1",
    document_type=DocumentType.WITHHOLDING_TAX_FORM,
    filename_markers=("국내원천소득", "제한세율"),
    text_markers=("국내원천소득 제한세율 적용신청서", "비거주자용"),
    quality_regions={
        "signature_present": RegionRule(0.73, 0.77, 0.95, 0.84, min_dark_ratio=0.02),
    },
    checkbox_regions={
        "all_no_boxes_checked": (
            RegionRule(0.92, 0.49, 0.96, 0.52, min_dark_ratio=0.02),
            RegionRule(0.92, 0.52, 0.96, 0.55, min_dark_ratio=0.02),
            RegionRule(0.92, 0.55, 0.96, 0.58, min_dark_ratio=0.02),
            RegionRule(0.92, 0.58, 0.96, 0.61, min_dark_ratio=0.02),
            RegionRule(0.92, 0.61, 0.96, 0.64, min_dark_ratio=0.02),
            RegionRule(0.92, 0.64, 0.96, 0.67, min_dark_ratio=0.02),
            RegionRule(0.92, 0.67, 0.96, 0.70, min_dark_ratio=0.02),
        ),
    },
    ocr_regions=(
        OCRRegionSpec("last_name", 0.14, 0.16, 0.30, 0.20),
        OCRRegionSpec("first_name", 0.40, 0.16, 0.54, 0.20),
        OCRRegionSpec("middle_name", 0.68, 0.16, 0.76, 0.20),
        OCRRegionSpec("address", 0.16, 0.21, 0.86, 0.25),
        OCRRegionSpec("tin", 0.14, 0.27, 0.31, 0.33),
        OCRRegionSpec("birth_date", 0.39, 0.27, 0.54, 0.33),
        OCRRegionSpec("phone_number", 0.13, 0.31, 0.39, 0.37),
        OCRRegionSpec("residency_country", 0.56, 0.27, 0.83, 0.33),
        OCRRegionSpec("residency_country_code", 0.86, 0.27, 0.98, 0.33),
        OCRRegionSpec("dividend_tax_rate", 0.72, 0.39, 0.90, 0.45),
        OCRRegionSpec("signature_date", 0.70, 0.74, 0.96, 0.81),
        OCRRegionSpec("applicant_name", 0.40, 0.79, 0.72, 0.85),
    ),
)


APOSTILLE_CALIFORNIA = DocumentTemplateProfile(
    template_id="apostille.california",
    document_type=DocumentType.APOSTILLE,
    filename_markers=("california",),
    text_markers=("state of california", "los angeles, california"),
    quality_regions={
        "seal_present": RegionRule(0.06, 0.74, 0.36, 0.97, min_dark_ratio=0.08),
        "signature_present": RegionRule(0.49, 0.78, 0.87, 0.93, min_dark_ratio=0.015),
    },
    ocr_regions=(
        OCRRegionSpec("issuing_country", 0.16, 0.34, 0.44, 0.40),
        OCRRegionSpec("signed_by", 0.12, 0.43, 0.40, 0.49),
        OCRRegionSpec("signer_capacity", 0.12, 0.50, 0.68, 0.57),
        OCRRegionSpec("seal_owner", 0.12, 0.57, 0.62, 0.63),
        OCRRegionSpec("issued_at", 0.12, 0.66, 0.42, 0.72),
        OCRRegionSpec("issued_on", 0.12, 0.72, 0.36, 0.78),
        OCRRegionSpec("issuing_authority", 0.12, 0.77, 0.72, 0.83),
        OCRRegionSpec("certificate_number", 0.07, 0.655, 0.20, 0.695),
    ),
)


APOSTILLE_MICHIGAN = DocumentTemplateProfile(
    template_id="apostille.michigan",
    document_type=DocumentType.APOSTILLE,
    filename_markers=("michigan",),
    text_markers=("state of michigan", "lansing, michigan"),
    quality_regions={
        "seal_present": RegionRule(0.06, 0.74, 0.38, 0.98, min_dark_ratio=0.08),
        "signature_present": RegionRule(0.55, 0.74, 0.90, 0.86, min_dark_ratio=0.02),
    },
    ocr_regions=(
        OCRRegionSpec("issuing_country", 0.34, 0.30, 0.80, 0.38),
        OCRRegionSpec("signed_by", 0.34, 0.37, 0.82, 0.46),
        OCRRegionSpec("signer_capacity", 0.43, 0.44, 0.72, 0.50),
        OCRRegionSpec("seal_owner", 0.42, 0.49, 0.79, 0.55),
        OCRRegionSpec("issued_at", 0.30, 0.57, 0.56, 0.62),
        OCRRegionSpec("issued_on", 0.40, 0.61, 0.66, 0.67),
        OCRRegionSpec("issuing_authority", 0.26, 0.68, 0.86, 0.74),
        OCRRegionSpec("certificate_number", 0.24, 0.73, 0.48, 0.81),
    ),
)


APOSTILLE_NORTH_CAROLINA = DocumentTemplateProfile(
    template_id="apostille.north_carolina",
    document_type=DocumentType.APOSTILLE,
    filename_markers=("california 주2", "north carolina"),
    text_markers=("state of north carolina", "raleigh, north carolina"),
    quality_regions={
        "seal_present": RegionRule(0.08, 0.68, 0.34, 0.96, min_dark_ratio=0.08),
        "signature_present": RegionRule(0.56, 0.70, 0.92, 0.92, min_dark_ratio=0.018),
    },
    ocr_regions=(
        OCRRegionSpec("issuing_country", 0.34, 0.31, 0.68, 0.36),
        OCRRegionSpec("signed_by", 0.36, 0.38, 0.72, 0.415),
        OCRRegionSpec("signer_capacity", 0.34, 0.41, 0.82, 0.46),
        OCRRegionSpec("seal_owner", 0.31, 0.44, 0.91, 0.49),
        OCRRegionSpec("issued_at", 0.13, 0.51, 0.57, 0.56),
        OCRRegionSpec("issued_on", 0.13, 0.55, 0.58, 0.60),
        OCRRegionSpec("issuing_authority", 0.12, 0.60, 0.93, 0.66),
        OCRRegionSpec("certificate_number", 0.08, 0.91, 0.48, 0.98),
    ),
)


PROFILES = {
    DocumentType.RESIDENCY_CERTIFICATE: (RESIDENCY_STANDARD,),
    DocumentType.WITHHOLDING_TAX_FORM: (WITHHOLDING_STANDARD,),
    DocumentType.APOSTILLE: (
        APOSTILLE_NORTH_CAROLINA,
        APOSTILLE_MICHIGAN,
        APOSTILLE_CALIFORNIA,
    ),
}


def profiles_for(document_type: DocumentType) -> tuple[DocumentTemplateProfile, ...]:
    return PROFILES.get(document_type, ())


def classify_template(
    document_type: DocumentType,
    source_path: str | Path,
    text: str | None = None,
) -> DocumentTemplateProfile | None:
    profiles = profiles_for(document_type)
    if not profiles:
        return None

    path_text = str(source_path).lower()
    text_lower = (text or "").lower()
    best_profile = profiles[0]
    best_score = -1
    for profile in profiles:
        score = 0
        score += sum(2 for marker in profile.filename_markers if marker.lower() in path_text)
        score += sum(3 for marker in profile.text_markers if marker.lower() in text_lower)
        if score > best_score:
            best_profile = profile
            best_score = score
    return best_profile


def find_profile_by_id(template_id: str | None) -> DocumentTemplateProfile | None:
    if not template_id:
        return None
    for profile_group in PROFILES.values():
        for profile in profile_group:
            if profile.template_id == template_id:
                return profile
    return None
