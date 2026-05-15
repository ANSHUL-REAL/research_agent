from src.document_loader import SourceDocument, dedupe_sources, prepare_documents
from src.text_splitter import chunk_documents


def test_prepare_documents_cleans_empty_and_duplicate_sources():
    sources = [
        SourceDocument(
            title="One",
            url="https://example.com/a",
            summary="",
            content="  Python\n\n\nRAG   guide  ",
            source_type="Blog",
        ),
        SourceDocument(
            title="Duplicate",
            url="https://example.com/a",
            summary="",
            content="Duplicate should disappear",
            source_type="Blog",
        ),
        SourceDocument(
            title="Empty",
            url="https://example.com/empty",
            summary="",
            content=" ",
            source_type="Blog",
        ),
    ]

    prepared = prepare_documents(sources, topic="RAG")

    assert len(prepared) == 1
    assert prepared[0].content == "Python RAG guide"
    assert prepared[0].metadata["topic"] == "RAG"
    assert prepared[0].metadata["source_url"] == "https://example.com/a"


def test_dedupe_sources_keeps_first_url():
    sources = [
        SourceDocument("A", "https://example.com", "", "alpha", "Blog"),
        SourceDocument("B", "https://example.com", "", "beta", "Blog"),
    ]

    assert dedupe_sources(sources)[0].title == "A"
    assert len(dedupe_sources(sources)) == 1


def test_chunk_documents_preserves_metadata_and_overlap():
    source = SourceDocument(
        title="Chunked",
        url="https://example.com/chunked",
        summary="",
        content="alpha beta gamma delta epsilon zeta eta theta",
        source_type="Blog",
        metadata={"source_title": "Chunked", "source_url": "https://example.com/chunked"},
    )

    chunks = chunk_documents([source], chunk_size=24, chunk_overlap=8)

    assert len(chunks) > 1
    assert chunks[0].metadata["source_title"] == "Chunked"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
