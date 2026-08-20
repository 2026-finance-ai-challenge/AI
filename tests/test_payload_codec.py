import json
from uuid import uuid4

import zstandard

from k_market_ai.rag.domain.models import SectionKind
from k_market_ai.rag.infrastructure.payload_codec import decode_sections


def test_decodes_backend_zstandard_payload() -> None:
    section_id = uuid4()
    document_id = uuid4()
    source = json.dumps(
        {
            "bodyText": "Revenue increased.",
            "sections": [
                {
                    "id": str(section_id),
                    "ordinal": 3,
                    "kind": "TEXT",
                    "heading": "Performance",
                    "text": "Revenue increased.",
                    "tableData": None,
                }
            ],
        }
    ).encode()
    compressed = zstandard.ZstdCompressor(level=6).compress(source)

    sections = decode_sections(compressed, document_id, 2)

    assert sections[0].id == section_id
    assert sections[0].document_id == document_id
    assert sections[0].document_version == 2
    assert sections[0].kind == SectionKind.TEXT
    assert sections[0].text == "Revenue increased."
