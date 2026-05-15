from langchain_core.documents import Document

from src.document_loader import SourceDocument
from src.evaluation import build_retrieval_diagnostics


def test_build_retrieval_diagnostics_calculates_visible_metrics():
    sources = [
        SourceDocument("Docs", "https://docs.example.com/rag", "", "alpha", "Official docs"),
        SourceDocument("Blog", "https://blog.example.com/rag", "", "beta", "Article"),
        SourceDocument("Repo", "https://github.com/example/rag", "", "gamma", "GitHub"),
    ]
    chunks = [
        Document(page_content="alpha" * 30, metadata={"source_url": "https://docs.example.com/rag"}),
        Document(page_content="beta" * 20, metadata={"source_url": "https://blog.example.com/rag"}),
    ]
    retrieved = [
        Document(page_content="alpha" * 30, metadata={"source_url": "https://docs.example.com/rag"}),
        Document(page_content="beta" * 20, metadata={"source_url": "https://blog.example.com/rag"}),
    ]
    citations = ["- [Docs](https://docs.example.com/rag)"]

    diagnostics = build_retrieval_diagnostics(sources, chunks, retrieved, citations)

    assert diagnostics.metrics["Sources"] == "3"
    assert diagnostics.metrics["Indexed chunks"] == "2"
    assert diagnostics.metrics["Retrieved chunks"] == "2"
    assert diagnostics.metrics["Citation coverage"] == "50%"
    assert diagnostics.metrics["Evidence accuracy"] == "50%"
    assert diagnostics.matrix[0]["Cited"] == 1
    assert diagnostics.matrix[0]["Retrieved but not cited"] == 1
