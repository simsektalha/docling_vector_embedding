from src.common.types import DocumentConversion, SectionText
from src.ingest.chunk import chunk_document


def test_chunk_hierarchical_basic():
    dc = DocumentConversion(
        doc_id="id",
        title=None,
        author=None,
        created_at=None,
        modified_at=None,
        language="en",
        sections=[SectionText(section_path="A", page_numbers=[], text="hello world")],
    )
    chunks = chunk_document(dc, strategy="hierarchical", max_tokens=5, overlap_tokens=2)
    assert len(chunks) >= 1
    assert all(c.doc_id == "id" for c in chunks)


def test_chunk_docling_uses_markdown():
    dc = DocumentConversion(
        doc_id="id",
        title=None,
        author=None,
        created_at=None,
        modified_at=None,
        language="en",
        markdown="# Title\n\n## Sub\ncontent\n\n## Another\nmore content",
        sections=[SectionText(section_path="A", page_numbers=[], text="fallback")],
    )
    chunks = chunk_document(dc, strategy="docling", max_tokens=20, overlap_tokens=5)
    assert len(chunks) >= 2


