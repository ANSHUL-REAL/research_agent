from __future__ import annotations

from langchain_core.documents import Document

from src.document_loader import SourceDocument


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def chunk_documents(
    sources: list[SourceDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    documents: list[Document] = []
    for source in sources:
        for index, chunk in enumerate(_split_text(source.content, chunk_size, chunk_overlap)):
            metadata = dict(source.metadata)
            metadata.setdefault("source_title", source.title)
            metadata.setdefault("source_url", source.url)
            metadata.setdefault("source_type", source.source_type)
            metadata["chunk_index"] = index
            documents.append(Document(page_content=chunk, metadata=metadata))
    return documents
