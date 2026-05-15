from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts import get_prompt_template
from src.utils import format_citations


class InsufficientContextError(RuntimeError):
    """Raised when retrieval returns no usable context."""


@dataclass(frozen=True)
class RagResult:
    answer: str
    citations: list[str]
    documents: list[Document]


def build_llm(model: str, google_api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=google_api_key,
        temperature=0.2,
    )


def render_context(documents: list[Document], max_chars: int = 12000) -> str:
    rendered: list[str] = []
    total = 0
    for index, document in enumerate(documents, start=1):
        title = document.metadata.get("source_title", "Source")
        url = document.metadata.get("source_url", "")
        block = f"[{index}] {title} - {url}\n{document.page_content.strip()}"
        if total + len(block) > max_chars:
            break
        rendered.append(block)
        total += len(block)
    return "\n\n".join(rendered)


def extract_response_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        if text_parts:
            return "".join(text_parts).strip()
    return str(content)


def clean_answer_markdown(answer: str) -> str:
    cleaned = re.split(r"\n\s*Sources\s*:\s*\n", answer.strip(), maxsplit=1, flags=re.IGNORECASE)[0].strip()
    heading_candidates = {
        "what is rag?",
        "what is rag",
        "when to use rag",
        "common use cases",
        "key takeaways",
        "missing topics",
        "corrected roadmap",
        "estimated timeline",
        "recommended resources",
        "project ideas",
        "learning plan",
        "timeline",
        "mini-projects",
        "final outcome",
    }
    lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        normalized = stripped.rstrip(":").lower()
        if (
            stripped
            and not stripped.startswith(("#", "-", "*", ">", "[", "|"))
            and normalized in heading_candidates
        ):
            lines.append(f"### {stripped.rstrip(':')}")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


class RagPipeline:
    def __init__(self, llm, retriever) -> None:
        self.llm = llm
        self.retriever = retriever

    def answer(self, question: str, mode: str) -> RagResult:
        documents = self.retriever.invoke(question)
        documents = [document for document in documents if document.page_content.strip()]
        if not documents:
            raise InsufficientContextError(
                "I could not find enough relevant context to answer this reliably."
            )

        prompt = get_prompt_template(mode)
        prompt_value = prompt.invoke(
            {
                "question": question,
                "context": render_context(documents),
            }
        )
        response = self.llm.invoke(prompt_value)
        answer = clean_answer_markdown(extract_response_text(response))
        return RagResult(
            answer=answer,
            citations=format_citations(document.metadata for document in documents),
            documents=documents,
        )
