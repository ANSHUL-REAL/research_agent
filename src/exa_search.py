from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from src.document_loader import SourceDocument, clean_text


def _read_attr(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _normalize_score(score: Any) -> float | None:
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def infer_source_type(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "github.com" in host:
        return "GitHub"
    if "arxiv.org" in host or path.endswith(".pdf"):
        return "Research paper"
    if "docs." in host or "/docs" in path or "developer." in host:
        return "Official docs"
    if "youtube.com" in host or "youtu.be" in host:
        return "Video"
    return "Article"


def normalize_exa_result(result: Any) -> SourceDocument:
    title = _read_attr(result, "title") or _read_attr(result, "url") or "Untitled source"
    url = _read_attr(result, "url") or ""
    text = _read_attr(result, "text") or _read_attr(result, "content") or ""
    highlights = _read_attr(result, "highlights") or []
    summary = _read_attr(result, "summary") or ""
    if not summary and highlights:
        summary = " ".join(str(item) for item in highlights[:2])
    score = _read_attr(result, "score")
    return SourceDocument(
        title=clean_text(str(title)),
        url=str(url),
        summary=clean_text(str(summary)),
        content=clean_text(str(text)),
        source_type=infer_source_type(str(url)),
        relevance_score=_normalize_score(score),
    )


def _source_rank(source: SourceDocument) -> tuple[int, float]:
    type_priority = {
        "Official docs": 0,
        "Research paper": 1,
        "GitHub": 2,
        "Article": 3,
        "Video": 4,
    }
    return (type_priority.get(source.source_type, 9), -(source.relevance_score or 0.0))


class ExaSearchClient:
    def __init__(self, api_key: str, exa_client: Any | None = None) -> None:
        if exa_client is None:
            from exa_py import Exa

            exa_client = Exa(api_key=api_key)
        self._client = exa_client

    def search(self, query: str, num_results: int = 5) -> list[SourceDocument]:
        try:
            response = self._client.search_and_contents(
                query,
                type="auto",
                num_results=num_results,
                text=True,
                highlights=True,
            )
        except Exception as exc:  # noqa: BLE001 - SDKs raise provider-specific exceptions.
            raise RuntimeError("Search failed. Please try again.") from exc

        results = getattr(response, "results", response.get("results", []) if isinstance(response, dict) else [])
        sources = [normalize_exa_result(result) for result in results]
        return sorted([source for source in sources if source.url], key=_source_rank)
