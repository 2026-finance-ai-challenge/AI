from __future__ import annotations

import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from k_market_ai.tax.document_model.schemas import OCRPage, OCRResult, OCRWordBox
from k_market_ai.tax.document_model.template_profiles import OCRRegionSpec

REGION_FALLBACK_VERTICAL_OFFSETS: dict[str, tuple[float, ...]] = {
    "issue_date": (-0.08, -0.06, -0.04, -0.02),
    "signed_by": (-0.14, -0.12, -0.10, -0.08),
}

REGION_FALLBACK_LEFT_OFFSETS: dict[str, tuple[float, ...]] = {
    "address": (-0.04, -0.06, -0.08),
}

REGION_FALLBACK_BOX_EXPANSIONS: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "address": (
        (0.0, 0.0, 0.0, 0.03),
        (0.0, -0.02, 0.0, 0.06),
    ),
    "applicant_name": ((-0.01, -0.01, 0.02, 0.01),),
    "middle_name": ((-0.02, -0.01, 0.03, 0.02),),
    "tin": (
        (0.0, 0.0, 0.03, 0.0),
        (0.0, 0.0, 0.06, 0.0),
        (0.0, 0.0, 0.0, 0.03),
        (0.0, 0.0, 0.06, 0.03),
    ),
}

REGION_SEARCH_ALL_FALLBACKS = {"address", "applicant_name", "middle_name", "tin"}
REGION_PREFER_LARGER_BOX_ON_TIE = {"middle_name"}

REGION_ALTERNATE_OVERRIDES: dict[str, tuple[dict[str, Any], ...]] = {
    "applicant_name": ({"lang": "en"},),
    "middle_name": ({"lang": "en"},),
}

REGION_PREFER_ALTERNATE_ENGINES = {"middle_name"}

MONTH_NAME_PATTERN = (
    r"\b(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b"
)
DATE_LIKE_REGIONS = {"issue_date", "signature_date", "issued_on"}


class _CheckpointRecognizer:
    def __init__(self, overrides: dict[str, Any]) -> None:
        self._paddleocr_home = Path(str(overrides["paddleocr_home"])).resolve()
        self._base_config = Path(str(overrides["base_config"])).resolve()
        self._checkpoint_path = Path(str(overrides["checkpoint_path"])).resolve()
        self._dict_path = Path(str(overrides["rec_char_dict_path"])).resolve()
        self._image_shape = str(overrides.get("rec_image_shape") or "3,48,320")
        self._max_text_length = int(overrides.get("max_text_length") or 32)
        self._model: Any = None
        self._ops: Any = None
        self._post_process: Any = None
        self._paddle: Any = None
        self._load()

    def _load(self) -> None:
        if str(self._paddleocr_home) not in sys.path:
            sys.path.insert(0, str(self._paddleocr_home))
        for module_name in list(sys.modules):
            if not (
                module_name == "ppocr"
                or module_name.startswith("ppocr.")
                or module_name == "tools"
                or module_name.startswith("tools.")
                or module_name == "ppstructure"
                or module_name.startswith("ppstructure.")
            ):
                continue
            module = sys.modules[module_name]
            module_file = getattr(module, "__file__", "") or ""
            if not str(module_file).startswith(str(self._paddleocr_home)):
                del sys.modules[module_name]

        import paddle
        from ppocr.data import create_operators, transform
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        from ppocr.utils.save_load import load_model
        from tools.program import load_config

        paddle.set_device("cpu")
        config = load_config(str(self._base_config))
        config["Global"]["use_gpu"] = False
        config["Global"]["pretrained_model"] = str(self._checkpoint_path)
        config["Global"]["character_dict_path"] = str(self._dict_path)
        config["Global"]["max_text_length"] = self._max_text_length
        config["Global"]["infer_img"] = self._image_shape

        post_process = build_post_process(config["PostProcess"], config["Global"])
        char_num = len(getattr(post_process, "character", []))
        if config["Architecture"]["Head"]["name"] == "MultiHead":
            out_channels_list = {
                "CTCLabelDecode": char_num,
                "SARLabelDecode": char_num + 2,
            }
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        elif char_num:
            config["Architecture"]["Head"]["out_channels"] = char_num

        model = build_model(config["Architecture"])
        load_model(config, model)
        model.eval()

        transforms: list[dict[str, Any]] = []
        for op in config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            if op_name == "RecResizeImg":
                op[op_name]["infer_mode"] = True
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            transforms.append(op)

        config["Global"]["infer_mode"] = True
        self._ops = create_operators(transforms, config["Global"])
        self._post_process = post_process
        self._model = model
        self._paddle = paddle
        self._transform = transform

    def ocr(self, image: str | Path | np.ndarray, cls: bool = True) -> list[list[object]]:
        del cls
        if isinstance(image, str | Path):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        batch = self._transform({"image": buffer.getvalue()}, self._ops)
        images = np.expand_dims(batch[0], axis=0)
        with self._paddle.no_grad():
            preds = self._model(self._paddle.to_tensor(images))
        post_result = self._post_process(preds)

        text = ""
        confidence = 0.0
        if isinstance(post_result, list) and post_result:
            payload = post_result[0]
            if isinstance(payload, list | tuple) and len(payload) >= 2:
                text = str(payload[0]).strip()
                confidence = float(payload[1])

        width, height = pil_image.size
        box = [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
        return [[[box, (text, confidence)]]]


class _TesseractOCR:
    def __init__(self, kwargs: dict[str, Any], overrides: dict[str, Any] | None = None) -> None:
        merged = {**kwargs, **(overrides or {})}
        self._lang = self._tesseract_lang(str(merged.get("lang") or "en"))
        self._config = str(
            merged.get("config")
            or os.getenv("KMARKET_TAX_OCR_TESSERACT_CONFIG")
            or "--oem 1 --psm 6"
        )
        self._pdf_dpi = int(merged.get("pdf_dpi") or os.getenv("KMARKET_TAX_OCR_PDF_DPI") or "250")
        if shutil.which("tesseract") is None:
            raise RuntimeError("tesseract runtime is missing. Rebuild the KART AI image.")

    def ocr(self, image: str | Path | np.ndarray, cls: bool = True) -> list[list[object]]:
        del cls
        pages = self._load_pages(image)
        raw_pages: list[list[object]] = []
        for page in pages:
            raw_pages.append(self._ocr_page(page))
        return raw_pages

    def _load_pages(self, image: str | Path | np.ndarray) -> list[Image.Image]:
        if isinstance(image, str | Path):
            path = Path(image)
            if path.suffix.lower() == ".pdf":
                try:
                    from pdf2image import convert_from_path
                except ImportError as exc:
                    raise RuntimeError("pdf2image runtime is missing.") from exc
                return [page.convert("RGB") for page in convert_from_path(path, dpi=self._pdf_dpi)]
            return [Image.open(path).convert("RGB")]
        return [Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")]

    def _ocr_page(self, page: Image.Image) -> list[object]:
        processed = self._preprocess(page)
        buffer = BytesIO()
        processed.save(buffer, format="PNG")
        # 작은 영역 OCR에서 OpenMP 스레드 경쟁을 없애고 호출별 실행 시간을 제한한다.
        result = subprocess.run(  # noqa: S603
            ["tesseract", "stdin", "stdout", "-l", self._lang, *shlex.split(self._config), "tsv"],  # noqa: S607
            input=buffer.getvalue(),
            capture_output=True,
            check=True,
            timeout=20,
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
        )
        rows = list(csv.DictReader(StringIO(result.stdout.decode("utf-8")), delimiter="\t"))
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
        for row in rows:
            if row.get("level") != "5" or not row.get("text", "").strip():
                continue
            key = tuple(row[field] for field in ("page_num", "block_num", "par_num", "line_num"))
            grouped.setdefault(key, []).append(row)
        lines: list[object] = []
        # 원래 모델의 줄 단위 결과를 유지해야 필드 파서가 불필요한 영역 재OCR을 하지 않는다.
        for words in grouped.values():
            normalized_text = " ".join(word["text"].strip() for word in words)
            confidences = [
                value for word in words if (value := self._confidence(word["conf"])) is not None
            ]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            left = min(float(word["left"]) for word in words)
            top = min(float(word["top"]) for word in words)
            right = max(float(word["left"]) + float(word["width"]) for word in words)
            bottom = max(float(word["top"]) + float(word["height"]) for word in words)
            box = [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ]
            lines.append([box, (normalized_text, confidence)])

        return lines

    def _preprocess(self, page: Image.Image) -> Image.Image:
        grayscale = ImageOps.grayscale(page)
        max_side = max(grayscale.size)
        if max_side < 1800:
            scale = max(2, int(1800 / max_side))
            grayscale = grayscale.resize(
                (grayscale.width * scale, grayscale.height * scale),
                resample=Image.Resampling.LANCZOS,
            )
        return grayscale.filter(ImageFilter.SHARPEN)

    def _confidence(self, raw_value: object) -> float | None:
        try:
            value = float(raw_value)
        except TypeError, ValueError:
            return None
        if value < 0:
            return None
        return round(value / 100.0, 4)

    def _tesseract_lang(self, lang: str) -> str:
        normalized = lang.lower().replace("-", "_")
        if normalized in {"ko", "kor", "korean", "korean+english", "kor+eng", "kor_eng"}:
            return "kor+eng"
        if normalized in {"eng", "en", "english"}:
            return "eng"
        return os.getenv("KMARKET_TAX_OCR_TESSERACT_LANG", "eng")


class TesseractOCREngine:
    """로컬 Tesseract 실행과 템플릿 영역 OCR을 제공한다."""

    def __init__(
        self,
        *,
        lang: str = "en",
        use_angle_cls: bool = True,
        show_log: bool = False,
        region_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self._kwargs = {
            "lang": lang,
            "use_angle_cls": use_angle_cls,
            "show_log": show_log,
            **kwargs,
        }
        self._region_overrides = region_overrides or {}
        self._engines: dict[tuple[tuple[str, str], ...], Any] = {}

    def _cache_key(self, overrides: dict[str, Any] | None = None) -> tuple[tuple[str, str], ...]:
        merged = {**self._kwargs, **(overrides or {})}
        return tuple(sorted((key, str(value)) for key, value in merged.items()))

    def _load(self, overrides: dict[str, Any] | None = None) -> Any:
        cache_key = self._cache_key(overrides)
        engine = self._engines.get(cache_key)
        if engine is None:
            if overrides and overrides.get("checkpoint_path"):
                engine = _CheckpointRecognizer(overrides)
            else:
                engine = _TesseractOCR(self._kwargs, overrides)
            self._engines[cache_key] = engine
        return engine

    def run(self, image_path: str | Path) -> OCRResult:
        engine = self._load()
        raw_result = engine.ocr(str(image_path), cls=self._kwargs.get("use_angle_cls", True))
        return OCRResult(pages=self._build_pages(raw_result))

    def run_regions(
        self,
        image_path: str | Path,
        region_specs: Iterable[OCRRegionSpec],
    ) -> dict[str, OCRPage]:
        path = Path(image_path)
        if path.suffix.lower() == ".pdf":
            return {}

        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            return {}

        regions: dict[str, OCRPage] = {}
        for region_spec in region_specs:
            region_engines = self._region_engines_for(region_spec.name)
            pages = self._run_region_with_fallbacks(
                image,
                region_spec,
                region_engines,
            )
            if pages:
                regions[region_spec.name] = pages[0]
        return regions

    def _run_region_with_fallbacks(
        self,
        image: Image.Image,
        region_spec: OCRRegionSpec,
        region_engines: list[Any],
    ) -> list[OCRPage]:
        region_boxes = [region_spec]
        for vertical_offset in REGION_FALLBACK_VERTICAL_OFFSETS.get(region_spec.name, ()):
            shifted_top = max(0.0, region_spec.top + vertical_offset)
            shifted_bottom = min(1.0, region_spec.bottom + vertical_offset)
            if shifted_bottom <= shifted_top:
                continue
            region_boxes.append(
                OCRRegionSpec(
                    region_spec.name,
                    region_spec.left,
                    shifted_top,
                    region_spec.right,
                    shifted_bottom,
                )
            )
        for left_offset in REGION_FALLBACK_LEFT_OFFSETS.get(region_spec.name, ()):
            shifted_left = max(0.0, region_spec.left + left_offset)
            if shifted_left >= region_spec.right:
                continue
            region_boxes.append(
                OCRRegionSpec(
                    region_spec.name,
                    shifted_left,
                    region_spec.top,
                    region_spec.right,
                    region_spec.bottom,
                )
            )
        for left_delta, top_delta, right_delta, bottom_delta in REGION_FALLBACK_BOX_EXPANSIONS.get(
            region_spec.name,
            (),
        ):
            shifted_left = max(0.0, region_spec.left + left_delta)
            shifted_top = max(0.0, region_spec.top + top_delta)
            shifted_right = min(1.0, region_spec.right + right_delta)
            shifted_bottom = min(1.0, region_spec.bottom + bottom_delta)
            if shifted_right <= shifted_left or shifted_bottom <= shifted_top:
                continue
            region_boxes.append(
                OCRRegionSpec(
                    region_spec.name,
                    shifted_left,
                    shifted_top,
                    shifted_right,
                    shifted_bottom,
                )
            )

        search_all_fallbacks = region_spec.name in REGION_SEARCH_ALL_FALLBACKS
        best_pages: list[OCRPage] = []
        best_score = (0, 0, 0)
        best_area = 0
        for candidate_region in region_boxes:
            crop = image.crop(
                (
                    int(image.width * candidate_region.left),
                    int(image.height * candidate_region.top),
                    int(image.width * candidate_region.right),
                    int(image.height * candidate_region.bottom),
                )
            )
            for variant in self._region_variants(region_spec.name, crop):
                for region_engine in region_engines:
                    raw_result = region_engine.ocr(
                        np.array(variant),
                        cls=self._kwargs.get("use_angle_cls", True),
                    )
                    pages = self._normalize_region_pages(
                        region_spec.name,
                        self._build_pages(raw_result),
                    )
                    score = self._score_region_pages(region_spec.name, pages)
                    crop_area = variant.width * variant.height
                    if score > best_score:
                        best_pages = pages
                        best_score = score
                        best_area = crop_area
                    elif (
                        region_spec.name in REGION_PREFER_LARGER_BOX_ON_TIE
                        and score == best_score
                        and crop_area > best_area
                    ):
                        best_pages = pages
                        best_area = crop_area
            if best_pages and not search_all_fallbacks:
                return best_pages
        return best_pages

    def _region_overrides_for(self, region_name: str) -> dict[str, Any] | None:
        override = self._region_overrides.get(region_name, {})
        return dict(override) or None

    def _region_engines_for(self, region_name: str) -> list[Any]:
        user_override = self._region_overrides_for(region_name)
        if user_override is not None:
            return [self._load(user_override)]

        if region_name in REGION_PREFER_ALTERNATE_ENGINES:
            alternate_engines = [
                self._load(override) for override in REGION_ALTERNATE_OVERRIDES.get(region_name, ())
            ]
            return [*alternate_engines, self._load(None)]

        engines = [self._load(None)]
        alternate_engines = [
            self._load(override) for override in REGION_ALTERNATE_OVERRIDES.get(region_name, ())
        ]
        engines.extend(alternate_engines)
        return engines

    def _region_variants(self, region_name: str, crop: Image.Image) -> list[Image.Image]:
        variants = [crop.convert("RGB")]
        if region_name in DATE_LIKE_REGIONS:
            grayscale = ImageOps.grayscale(crop)
            variants.extend(
                [
                    crop.resize((crop.width * 2, crop.height * 2)).convert("RGB"),
                    grayscale.resize((crop.width * 4, crop.height * 4)).convert("RGB"),
                    grayscale.point(lambda p: 255 if p > 180 else 0)
                    .resize((crop.width * 4, crop.height * 4))
                    .convert("RGB"),
                    crop.filter(ImageFilter.SHARPEN)
                    .resize((crop.width * 4, crop.height * 4))
                    .convert("RGB"),
                ]
            )
        elif region_name == "certificate_number":
            grayscale = ImageOps.grayscale(crop)
            variants = [
                grayscale.filter(ImageFilter.SHARPEN)
                .resize((crop.width * 4, crop.height * 4))
                .convert("RGB"),
                grayscale.point(lambda p: 255 if p > 180 else 0)
                .resize((crop.width * 4, crop.height * 4))
                .convert("RGB"),
                crop.resize((crop.width * 2, crop.height * 2)).convert("RGB"),
                crop.convert("RGB"),
            ]
        elif region_name == "applicant_name":
            grayscale = ImageOps.grayscale(crop)
            variants = [
                crop.convert("RGB"),
                crop.resize((crop.width * 2, crop.height * 2)).convert("RGB"),
                grayscale.resize((crop.width * 4, crop.height * 4)).convert("RGB"),
                crop.filter(ImageFilter.SHARPEN)
                .resize((crop.width * 4, crop.height * 4))
                .convert("RGB"),
            ]
        elif region_name == "middle_name":
            grayscale = ImageOps.grayscale(crop)
            variants = [
                crop.convert("RGB"),
                crop.resize((crop.width * 2, crop.height * 2)).convert("RGB"),
                grayscale.resize((crop.width * 4, crop.height * 4)).convert("RGB"),
                grayscale.point(lambda p: 255 if p > 180 else 0)
                .resize(
                    (crop.width * 4, crop.height * 4),
                    resample=Image.Resampling.NEAREST,
                )
                .convert("RGB"),
                grayscale.point(lambda p: 255 if p > 200 else 0)
                .resize(
                    (crop.width * 4, crop.height * 4),
                    resample=Image.Resampling.NEAREST,
                )
                .convert("RGB"),
                crop.filter(ImageFilter.SHARPEN)
                .resize((crop.width * 4, crop.height * 4))
                .convert("RGB"),
            ]
        elif region_name == "tin":
            grayscale = ImageOps.grayscale(crop)
            variants = [
                crop.convert("RGB"),
                crop.resize((crop.width * 2, crop.height * 2)).convert("RGB"),
                grayscale.resize((crop.width * 4, crop.height * 4)).convert("RGB"),
                crop.filter(ImageFilter.SHARPEN)
                .resize((crop.width * 4, crop.height * 4))
                .convert("RGB"),
            ]
        return variants

    def _score_region_pages(
        self,
        region_name: str,
        pages: list[OCRPage],
    ) -> tuple[int, int, int]:
        text = "\n".join(page.raw_text for page in pages if page.raw_text).strip()
        if not text:
            return (0, 0, 0)
        normalized = re.sub(r"\s+", " ", text).strip()

        if region_name == "certificate_number":
            for line in text.splitlines():
                normalized_line = re.sub(r"\s+", " ", line).strip()
                if "no" not in normalized_line.lower():
                    continue
                match = re.search(r"\bNo\.?\s*([A-Z0-9-]+)\b", normalized_line, re.IGNORECASE)
                if match and match.group(1) not in {"8", "9", "10"}:
                    return (4, len(match.group(1)), -len(normalized_line))
                return (2, len(normalized_line), -len(normalized_line))

            digits = [
                token
                for token in re.findall(r"\b\d+\b", normalized)
                if token not in {"8", "9", "10"}
            ]
            if digits:
                return (3, len(max(digits, key=len)), -len(normalized))
            return (1, len(normalized), -len(normalized))

        if region_name == "tin":
            normalized_compact = re.sub(r"\s+", "", normalized)
            if re.search(r"\b\d{3}-\d{2}-\d{4}\b", normalized):
                return (5, 9, -len(normalized))
            if re.search(r"\b\d{2}-\d{7}\b", normalized):
                return (5, 9, -len(normalized))
            if re.search(r"\b\d{3}-\d{2}-\d{3}\b", normalized_compact):
                return (3, 8, -len(normalized))
            if re.search(r"\b\d{3}-\d{5}\b", normalized_compact):
                return (2, 8, -len(normalized))
            if re.search(r"\b\d{3}-\d{2}-\d{2}\b", normalized_compact):
                return (2, 7, -len(normalized))
            digit_count = len(re.findall(r"\d", normalized))
            return (1, digit_count, -len(normalized))

        if region_name == "issue_date":
            has_year = re.search(r"\b\d{4}\b", normalized) is not None
            if re.search(MONTH_NAME_PATTERN, normalized, re.IGNORECASE) and has_year:
                return (4, len(normalized), -len(normalized))
            if "date" in normalized.lower() and has_year:
                return (3, len(normalized), -len(normalized))
            if has_year:
                return (2, len(normalized), -len(normalized))

        if region_name == "signature_date":
            normalized_compact = re.sub(r"\s+", "", normalized)
            if re.search(r"\b\d{4}-\d{2}-\d{2}\b", normalized):
                return (5, len(normalized), -len(normalized))
            if re.search(r"\b\d{4}\.\d{2}\.\d{2}\b", normalized):
                return (4, len(normalized), -len(normalized))
            if re.search(r"\b\d{4}년\d{1,2}월\d{1,2}일\b", normalized_compact):
                return (4, len(normalized), -len(normalized))
            if re.search(r"\b\d{4}\b", normalized):
                return (2, len(normalized), -len(normalized))

        if region_name == "issued_on":
            has_year = re.search(r"\b\d{4}\b", normalized) is not None
            normalized_upper = normalized.upper()
            if has_year and re.search(r"\bDAY OF\b", normalized_upper):
                return (5, len(normalized), -len(normalized))
            if has_year and re.search(MONTH_NAME_PATTERN, normalized, re.IGNORECASE):
                return (4, len(normalized), -len(normalized))
            if has_year:
                return (2, len(normalized), -len(normalized))

        if region_name == "signed_by":
            normalized_lower = normalized.lower()
            if re.fullmatch(r"\d+\.?", normalized):
                return (0, 0, 0)
            alpha_token_count = len(re.findall(r"[A-Za-z]+", normalized))
            score = 1
            if alpha_token_count >= 2:
                score += 2
            if re.search(r"\b(sample|notary)\b", normalized_lower):
                score += 1
            if re.search(
                r"(acting in the capacity|secretary of state|bears the seal|country:)",
                normalized_lower,
            ):
                score -= 2
            return (score, alpha_token_count, -len(normalized))

        if region_name == "address":
            normalized_lower = normalized.lower()
            has_leading_number = re.match(r"^\d{1,5}\b", normalized) is not None
            has_country_tail = "united states" in normalized_lower or normalized_lower.endswith(
                "usa"
            )
            has_street_marker = bool(
                re.search(
                    r"\b(street|st|road|rd|avenue|ave|blvd|boulevard|suite|apt)\b",
                    normalized_lower,
                )
            )
            score = 1
            if has_country_tail:
                score += 1
            if has_street_marker:
                score += 1
            if has_leading_number:
                score += 2
            return (score, len(normalized), -len(normalized))

        if region_name == "applicant_name":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            tokens = re.findall(r"[A-Za-z0-9'.-]+", normalized)
            name_like_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
            digit_only_tokens = [token for token in tokens if token.isdigit()]
            score = 0
            if 2 <= len(name_like_tokens) <= 3:
                score += 3
            elif len(name_like_tokens) > 3:
                score += 1
            if len(tokens) == len(name_like_tokens):
                score += 1
            else:
                score -= len(tokens) - len(name_like_tokens)
            if len(lines) >= 2:
                score += 1
            if any(token.upper() == "USER" for token in name_like_tokens):
                score += 1
            if any(
                len(token.strip(".")) == 1 and token.strip(".").isalpha()
                for token in name_like_tokens
            ):
                score += 1
            if any(len(token) >= 3 for token in digit_only_tokens):
                score -= 2
            if re.search(r"\b(name|last|first|middle)\b", normalized.lower()):
                score -= 2
            return (score, len(name_like_tokens), -len(tokens))

        if region_name == "middle_name":
            tokens = re.findall(r"[A-Za-z0-9]+", normalized)
            if not tokens:
                return (0, 0, 0)
            token = max(tokens, key=len)
            score = 1
            if len(token) == 1:
                score += 2
            if token.isalpha():
                score += 2
            elif token.isdigit():
                score += 1
            return (score, int(token.isalpha()), -len(normalized))

        return (1, len(normalized), -len(normalized))

    def _normalize_region_pages(
        self,
        region_name: str,
        pages: list[OCRPage],
    ) -> list[OCRPage]:
        if region_name == "applicant_name":
            best_line = self._select_best_applicant_name_line(pages)
        elif region_name == "address":
            best_line = self._select_best_address_line(pages)
        elif region_name == "middle_name":
            best_line = self._select_best_middle_name_line(pages)
        else:
            return pages
        if best_line is None:
            return pages
        page_number, raw_text = best_line
        normalized_pages: list[OCRPage] = []
        for page in pages:
            if page.page_number != page_number:
                continue
            normalized_pages.append(
                OCRPage(
                    page_number=page.page_number,
                    words=page.words,
                    raw_text=raw_text,
                )
            )
        return normalized_pages or pages

    def _select_best_applicant_name_line(
        self,
        pages: list[OCRPage],
    ) -> tuple[int, str] | None:
        best_candidate: tuple[tuple[int, int, int], int, str] | None = None
        for page in pages:
            for line in page.raw_text.splitlines():
                normalized_line = re.sub(r"\s+", " ", line).strip()
                if not normalized_line:
                    continue
                score = self._score_applicant_name_line(normalized_line)
                if score is None:
                    continue
                candidate = (score, page.page_number, normalized_line)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate
        if best_candidate is None:
            return None
        score, page_number, normalized_line = best_candidate
        if score[0] < 5 or score[1] < 2:
            return None
        return page_number, normalized_line

    def _select_best_address_line(
        self,
        pages: list[OCRPage],
    ) -> tuple[int, str] | None:
        best_candidate: tuple[tuple[int, int, int], int, str] | None = None
        for page in pages:
            for line in page.raw_text.splitlines():
                normalized_line = re.sub(r"\s+", " ", line).strip()
                if not normalized_line:
                    continue
                score = self._score_address_line(normalized_line)
                if score is None:
                    continue
                candidate = (score, page.page_number, normalized_line)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate
        if best_candidate is None:
            return None
        score, page_number, normalized_line = best_candidate
        if score[0] < 6:
            return None
        return page_number, normalized_line

    def _select_best_middle_name_line(
        self,
        pages: list[OCRPage],
    ) -> tuple[int, str] | None:
        best_candidate: tuple[tuple[int, int, int], int, str] | None = None
        for page in pages:
            for line in page.raw_text.splitlines():
                normalized_line = re.sub(r"\s+", " ", line).strip()
                if not normalized_line:
                    continue
                score = self._score_middle_name_line(normalized_line)
                if score is None:
                    continue
                token = re.sub(r"[^A-Za-z0-9]", "", normalized_line)
                if not token:
                    continue
                candidate = (score, page.page_number, token)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate
        if best_candidate is None:
            return None
        score, page_number, token = best_candidate
        if score[0] < 4:
            return None
        return page_number, token

    def _score_address_line(
        self,
        normalized_line: str,
    ) -> tuple[int, int, int] | None:
        normalized_lower = normalized_line.lower()
        tokens = re.findall(r"[A-Za-z0-9#.'-]+", normalized_line)
        if not tokens:
            return None

        score = 0
        if re.match(r"^\d{1,5}\b", normalized_line):
            score += 3
        if re.search(
            r"\b(street|st|road|rd|avenue|ave|blvd|boulevard|suite|apt)\b",
            normalized_lower,
        ):
            score += 4
        if "united states" in normalized_lower or normalized_lower.endswith("usa"):
            score += 1
        if re.search(r"\b\d{5}\b", normalized_line):
            score += 1
        if len(tokens) >= 6:
            score += 1
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", normalized_line):
            score -= 4
        if re.search(r"\buser\b", normalized_lower):
            score -= 2
        return (score, len(tokens), -len(normalized_line))

    def _score_middle_name_line(
        self,
        normalized_line: str,
    ) -> tuple[int, int, int] | None:
        token = re.sub(r"[^A-Za-z0-9]", "", normalized_line)
        if not token:
            return None

        score = 0
        if re.fullmatch(r"[A-Za-z]", normalized_line):
            score += 5
        elif len(token) == 1 and token.isalpha():
            score += 3
        elif token.isalpha() or token.isdigit():
            score += 1

        if re.search(r"\b(name|middle)\b", normalized_line.lower()):
            score -= 2
        noise_count = len(re.sub(r"[A-Za-z0-9\s]", "", normalized_line))
        return (score, int(token.isalpha()), -noise_count)

    def _score_applicant_name_line(
        self,
        normalized_line: str,
    ) -> tuple[int, int, int] | None:
        tokens = re.findall(r"[A-Za-z0-9'.-]+", normalized_line)
        if not tokens:
            return None
        alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
        if len(alpha_tokens) < 2 or len(alpha_tokens) > 3:
            return None

        digit_only_tokens = [token for token in tokens if token.isdigit()]
        long_alpha_tokens = [
            token for token in alpha_tokens if len(token.strip(".").replace("-", "")) >= 3
        ]
        middle_initial_count = sum(
            1 for token in alpha_tokens if len(token.strip(".")) == 1 and token.strip(".").isalpha()
        )

        score = 0
        if len(alpha_tokens) == 3 and middle_initial_count >= 1 and len(long_alpha_tokens) >= 2:
            score += 6
        elif len(alpha_tokens) == 2 and len(long_alpha_tokens) >= 2:
            score += 5
        elif len(alpha_tokens) == 3 and len(long_alpha_tokens) >= 2:
            score += 3
        else:
            score += 1

        if len(tokens) == len(alpha_tokens):
            score += 1
        else:
            score -= len(tokens) - len(alpha_tokens)
        if any(token.upper() == "USER" for token in alpha_tokens):
            score += 1
        if any(len(token) >= 3 for token in digit_only_tokens):
            score -= 2
        if re.search(r"\b(name|last|first|middle)\b", normalized_line.lower()):
            score -= 2

        return (score, len(long_alpha_tokens), -len(tokens))

    def _build_pages(self, raw_result: Any) -> list[OCRPage]:
        pages: list[OCRPage] = []
        for page_number, page_lines in enumerate(raw_result or [], start=1):
            words: list[OCRWordBox] = []
            text_chunks: list[str] = []
            for line in page_lines or []:
                if len(line) < 2:
                    continue
                box, payload = line
                text = str(payload[0]).strip()
                confidence = float(payload[1]) if len(payload) > 1 else None
                if text:
                    text_chunks.append(text)
                    words.append(OCRWordBox(text=text, confidence=confidence, points=box))
            pages.append(
                OCRPage(
                    page_number=page_number,
                    words=words,
                    raw_text="\n".join(text_chunks),
                )
            )
        return pages
