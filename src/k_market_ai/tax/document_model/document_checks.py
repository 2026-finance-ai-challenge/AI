from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from k_market_ai.tax.document_model.schemas import DocumentType
from k_market_ai.tax.document_model.template_profiles import (
    RegionRule,
    classify_template,
    find_profile_by_id,
)


def compute_document_checks(
    document_type: DocumentType,
    source_path: str | Path,
    *,
    template_id: str | None = None,
    ocr_text: str | None = None,
) -> dict[str, bool | int | None]:
    image_array = _load_grayscale_image(source_path)
    if image_array is None:
        return {
            "seal_present": None,
            "signature_present": None,
            "all_no_boxes_checked": None,
            "checked_no_box_count": None,
        }

    profile = find_profile_by_id(template_id) or classify_template(
        document_type,
        source_path,
        ocr_text,
    )
    if profile is None:
        return {}

    checks = _evaluate_region_map(image_array, profile.quality_regions)
    for field_name, rules in profile.checkbox_regions.items():
        checked = [_region_has_mark(image_array, rule) for rule in rules]
        checks["checked_no_box_count"] = sum(1 for flag in checked if flag)
        checks[field_name] = all(checked)
    return checks


def _evaluate_region_map(
    image_array: np.ndarray,
    region_map: dict[str, RegionRule],
) -> dict[str, bool]:
    return {
        field_name: _region_has_mark(image_array, rule) for field_name, rule in region_map.items()
    }


def _region_has_mark(image_array: np.ndarray, rule: RegionRule) -> bool:
    cropped = _crop_region(image_array, rule)
    if cropped.size == 0:
        return False
    dark_ratio = float(np.mean(cropped < rule.dark_threshold))
    return dark_ratio >= rule.min_dark_ratio


def _crop_region(image_array: np.ndarray, rule: RegionRule) -> np.ndarray:
    height, width = image_array.shape
    left = max(0, min(width - 1, int(width * rule.left)))
    top = max(0, min(height - 1, int(height * rule.top)))
    right = max(left + 1, min(width, int(width * rule.right)))
    bottom = max(top + 1, min(height, int(height * rule.bottom)))
    return image_array[top:bottom, left:right]


def _load_grayscale_image(source_path: str | Path) -> np.ndarray | None:
    path = Path(source_path)
    if path.suffix.lower() == ".pdf":
        return None
    try:
        image = Image.open(path).convert("L")
    except OSError:
        return None
    return np.array(image, dtype=np.uint8)
