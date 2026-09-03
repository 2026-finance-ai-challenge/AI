import hashlib
import json
from typing import Any

from k_market_ai.rag.domain.chunker import normalize_text
from k_market_ai.rag.domain.models import SourceSection


def translated_selection_exists(
    section: SourceSection,
    selected: str,
    source_hash: str,
    canonical_source: str,
    result: Any,
) -> bool:
    if not selected or hashlib.sha256(canonical_source.encode()).hexdigest() != source_hash:
        return False
    try:
        source = json.loads(canonical_source)
    except ValueError, TypeError:
        return False
    # 공시의 현재 원문과 일치하는 완료 캐시만 선택 근거로 인정한다.
    if not isinstance(source, dict) or source.get("text") != section.text:
        return False
    if source.get("heading") != section.heading or not isinstance(result, dict):
        return False
    text = result.get("translatedText")
    if isinstance(text, str) and selected in normalize_text(text):
        return True
    table = result.get("translatedTableData")
    if not isinstance(table, list) or not all(isinstance(row, list) for row in table):
        return False
    cells = [cell for row in table for cell in row]
    if not all(isinstance(cell, str) for cell in cells):
        return False
    return selected in normalize_text(" ".join(cells))
