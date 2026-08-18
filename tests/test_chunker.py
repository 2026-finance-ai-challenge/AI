from uuid import UUID, uuid4

from k_market_ai.rag.domain.chunker import MAX_CHARS, chunk_sections
from k_market_ai.rag.domain.models import SectionKind, SourceSection


def test_chunker_preserves_document_and_section_boundaries() -> None:
    document_id = uuid4()
    sections = [
        SourceSection(
            id=uuid4(),
            document_id=document_id,
            document_version=2,
            ordinal=0,
            kind=SectionKind.TITLE,
            heading="Business overview",
            text="Business overview",
        ),
        SourceSection(
            id=uuid4(),
            document_id=document_id,
            document_version=2,
            ordinal=1,
            kind=SectionKind.TEXT,
            heading=None,
            text="A" * (MAX_CHARS + 100),
        ),
    ]

    chunks = chunk_sections(sections)

    assert len(chunks) == 3
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.document_id == document_id for chunk in chunks)
    assert all(chunk.document_version == 2 for chunk in chunks)
    assert all(chunk.content_hash for chunk in chunks)


def test_chunker_does_not_join_different_documents() -> None:
    first_document = uuid4()
    second_document = uuid4()
    sections = [
        _section(first_document, 0, "first"),
        _section(second_document, 0, "second"),
    ]

    chunks = chunk_sections(sections)

    assert len(chunks) == 2
    assert chunks[0].document_id == first_document
    assert chunks[1].document_id == second_document


def _section(document_id: UUID, ordinal: int, text: str) -> SourceSection:
    return SourceSection(
        id=uuid4(),
        document_id=document_id,
        document_version=1,
        ordinal=ordinal,
        kind=SectionKind.TEXT,
        heading=None,
        text=text,
    )
