from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from langchain_core.documents import Document

from src.document_loader import SourceDocument


@dataclass(frozen=True)
class RetrievalDiagnostics:
    metrics: dict[str, str]
    matrix: list[dict[str, int | str]]
    source_type_rows: list[dict[str, int | str]]


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _urls_from_citations(citations: list[str]) -> set[str]:
    urls: set[str] = set()
    for citation in citations:
        if "](" not in citation:
            continue
        url = citation.split("](", 1)[1].rstrip(")")
        urls.add(url)
    return urls


def _percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0%"
    return f"{round((numerator / denominator) * 100)}%"


def build_retrieval_diagnostics(
    sources: list[SourceDocument],
    chunks: list[Document],
    retrieved_documents: list[Document],
    citations: list[str],
) -> RetrievalDiagnostics:
    source_urls = {source.url for source in sources}
    retrieved_urls = {
        document.metadata.get("source_url", "")
        for document in retrieved_documents
        if document.metadata.get("source_url")
    }
    citation_urls = _urls_from_citations(citations)
    cited_retrieved = len(retrieved_urls & citation_urls)
    retrieved_not_cited = max(len(retrieved_urls - citation_urls), 0)
    not_retrieved = max(len(source_urls - retrieved_urls), 0)
    total_chars = sum(len(document.page_content) for document in chunks)
    avg_chunk_chars = round(total_chars / len(chunks)) if chunks else 0
    source_domains = {_domain(source.url) for source in sources if source.url}
    source_types: dict[str, int] = {}
    for source in sources:
        source_types[source.source_type] = source_types.get(source.source_type, 0) + 1

    metrics = {
        "Sources": str(len(sources)),
        "Indexed chunks": str(len(chunks)),
        "Retrieved chunks": str(len(retrieved_documents)),
        "Source diversity": str(len(source_domains)),
        "Citation coverage": _percent(cited_retrieved, len(retrieved_urls)),
        "Evidence accuracy": _percent(cited_retrieved, max(len(retrieved_urls | citation_urls), 1)),
        "Avg chunk size": f"{avg_chunk_chars} chars",
    }
    matrix = [
        {
            "Prediction": "Retrieved as relevant",
            "Cited": cited_retrieved,
            "Retrieved but not cited": retrieved_not_cited,
        },
        {
            "Prediction": "Not retrieved",
            "Cited": max(len(citation_urls - retrieved_urls), 0),
            "Retrieved but not cited": not_retrieved,
        },
    ]
    source_type_rows = [
        {"Source type": source_type, "Count": count}
        for source_type, count in sorted(source_types.items())
    ]
    return RetrievalDiagnostics(metrics=metrics, matrix=matrix, source_type_rows=source_type_rows)
