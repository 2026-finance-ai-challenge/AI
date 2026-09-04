import json
from uuid import uuid4

import zstandard

from k_market_ai.rag.domain.chunker import chunk_sections
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


def test_table_cells_are_preserved_without_changing_indexed_text() -> None:
    document_id = uuid4()
    section = {
        "id": str(uuid4()),
        "ordinal": 1,
        "kind": "TABLE",
        "heading": None,
        "text": "기간 당기 전기 매출액 100 90",
    }
    rows = [["기간", "당기", "전기"], ["매출액", "100", "90"]]

    def decode(value):
        payload = json.dumps({"sections": [{**section, "tableData": value}]}).encode()
        return decode_sections(zstandard.ZstdCompressor().compress(payload), document_id, 1)

    baseline = decode(None)
    structured = decode(json.dumps(rows))
    assert structured[0].table_rows == tuple(tuple(row) for row in rows)
    assert chunk_sections(structured) == chunk_sections(baseline)
    assert decode("invalid")[0].table_rows == ()
