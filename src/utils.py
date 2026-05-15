from __future__ import annotations

from collections.abc import Iterable


def format_citations(metadata_items: Iterable[dict]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for metadata in metadata_items:
        title = metadata.get("source_title") or "Source"
        url = metadata.get("source_url") or ""
        key = f"{title}|{url}"
        if not url or key in seen:
            continue
        seen.add(key)
        citations.append(f"- [{title}]({url})")
    return citations


def build_markdown_report(query: str, mode: str, answer: str, citations: list[str]) -> str:
    source_block = "\n".join(citations) if citations else "No citations available."
    return f"""# AI Roadmap & Project Research Copilot Report

**Mode:** {mode}

**Query:**
{query}

## Answer

{answer}

## Sources

{source_block}
"""


def source_card_summary(summary: str, content: str, limit: int = 240) -> str:
    text = (summary or content or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
