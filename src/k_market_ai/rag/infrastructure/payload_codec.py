import json
from io import BytesIO
from typing import Any
from uuid import UUID

import zstandard

from k_market_ai.rag.domain.models import SectionKind, SourceSection

MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024


def decode_sections(
    compressed: bytes | memoryview,
    document_id: UUID,
    document_version: int,
) -> list[SourceSection]:
    with zstandard.ZstdDecompressor().stream_reader(BytesIO(compressed)) as reader:
        source = reader.read(MAX_DECOMPRESSED_BYTES + 1)
    if len(source) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("Disclosure payload exceeds decompression limit")
    payload: dict[str, Any] = json.loads(source)
    return [
        SourceSection(
            id=UUID(str(section["id"])),
            document_id=document_id,
            document_version=document_version,
            ordinal=int(section["ordinal"]),
            kind=SectionKind(str(section["kind"])),
            heading=None if section.get("heading") is None else str(section["heading"]),
            text=str(section["text"]),
        )
        for section in payload["sections"]
    ]
