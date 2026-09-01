from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def compute_quality_metrics(
    source_path: str | Path,
    *,
    blur_threshold: float = 100.0,
) -> dict[str, float | bool | int | str | None]:
    path = Path(source_path)
    metrics: dict[str, float | bool | int | str | None] = {
        "file_type": path.suffix.lower(),
        "width": None,
        "height": None,
        "blur_score": None,
        "blurry": None,
    }

    if path.suffix.lower() == ".pdf":
        return metrics

    try:
        image = Image.open(path).convert("L")
    except OSError:
        return metrics

    width, height = image.size
    grayscale = np.array(image, dtype=np.float32)
    laplacian = (
        -4.0 * grayscale
        + np.roll(grayscale, 1, axis=0)
        + np.roll(grayscale, -1, axis=0)
        + np.roll(grayscale, 1, axis=1)
        + np.roll(grayscale, -1, axis=1)
    )
    blur_score = float(laplacian.var())

    metrics.update(
        {
            "width": int(width),
            "height": int(height),
            "blur_score": blur_score,
            "blurry": blur_score < blur_threshold,
        }
    )
    return metrics


def average_ocr_confidence(pages: list[dict[str, object]] | list[object]) -> float | None:
    confidences: list[float] = []
    for page in pages:
        words = getattr(page, "words", None)
        if words is None and isinstance(page, dict):
            words = page.get("words")
        if not words:
            continue
        for word in words:
            confidence = getattr(word, "confidence", None)
            if confidence is None and isinstance(word, dict):
                confidence = word.get("confidence")
            if confidence is not None:
                confidences.append(float(confidence))
    if not confidences:
        return None
    return float(np.mean(confidences))
