from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import requests
from bs4 import BeautifulSoup


@dataclass
class SourceDocument:
    title: str
    url: str
    summary: str
    content: str
    source_type: str
    relevance_score: float | None = None
    metadata: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def dedupe_sources(sources: Iterable[SourceDocument]) -> list[SourceDocument]:
    seen: set[str] = set()
    unique: list[SourceDocument] = []
    for source in sources:
        normalized_url = source.url.rstrip("/")
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        unique.append(source)
    return unique


def fetch_url_text(url: str, timeout: int = 12) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "AI-Roadmap-Research-Copilot/0.1"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))


def hydrate_missing_content(sources: Iterable[SourceDocument]) -> list[SourceDocument]:
    hydrated: list[SourceDocument] = []
    for source in sources:
        if clean_text(source.content):
            hydrated.append(source)
            continue
        try:
            source.content = fetch_url_text(source.url)
        except requests.RequestException:
            source.content = ""
        hydrated.append(source)
    return hydrated


def prepare_documents(sources: Iterable[SourceDocument], topic: str) -> list[SourceDocument]:
    prepared: list[SourceDocument] = []
    created_at = datetime.now(timezone.utc).isoformat()
    for source in dedupe_sources(sources):
        content = clean_text(source.content)
        if not content:
            continue
        metadata = {
            "source_title": source.title or source.url,
            "source_url": source.url,
            "source_type": source.source_type,
            "topic": topic,
            "created_at": created_at,
        }
        if source.relevance_score is not None:
            metadata["relevance_score"] = source.relevance_score
        metadata.update(source.metadata)
        prepared.append(
            SourceDocument(
                title=source.title or source.url,
                url=source.url,
                summary=clean_text(source.summary),
                content=content,
                source_type=source.source_type,
                relevance_score=source.relevance_score,
                metadata=metadata,
            )
        )
    return prepared
