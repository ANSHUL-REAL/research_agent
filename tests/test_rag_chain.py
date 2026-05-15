from langchain_core.documents import Document

from src.rag_chain import (
    InsufficientContextError,
    RagPipeline,
    clean_answer_markdown,
    extract_response_text,
    render_context,
)


class DummyRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        assert query == "What is RAG?"
        return self.docs


class DummyLlm:
    def invoke(self, prompt_value):
        text = prompt_value.to_string()
        assert "Retrieved context" in text
        return type("Response", (), {"content": "RAG uses retrieved context. [Docs](https://docs.example.com)"})()


def test_extract_response_text_handles_gemini_content_blocks():
    response = type(
        "Response",
        (),
        {
            "content": [
                {"type": "text", "text": "### Clean answer\n"},
                {"type": "text", "text": "Only this text should render."},
                {"type": "metadata", "extras": {"signature": "hidden"}},
            ]
        },
    )()

    assert extract_response_text(response) == "### Clean answer\nOnly this text should render."


def test_clean_answer_markdown_removes_model_generated_sources_section():
    answer = """RAG uses retrieved context.

Sources:

[1] Docs - https://docs.example.com
[2] Blog - https://blog.example.com
"""

    assert clean_answer_markdown(answer) == "RAG uses retrieved context."


def test_clean_answer_markdown_promotes_common_plain_headings():
    answer = """What is RAG?
RAG retrieves context.

When to Use RAG
- When knowledge changes.
"""

    assert clean_answer_markdown(answer).startswith("### What is RAG?\n")
    assert "\n### When to Use RAG\n" in clean_answer_markdown(answer)


def test_render_context_includes_numbered_sources():
    docs = [
        Document(
            page_content="RAG retrieves relevant chunks.",
            metadata={"source_title": "Docs", "source_url": "https://docs.example.com"},
        )
    ]

    context = render_context(docs)

    assert "[1] Docs - https://docs.example.com" in context
    assert "RAG retrieves relevant chunks." in context


def test_rag_pipeline_generates_answer_and_citations():
    docs = [
        Document(
            page_content="RAG retrieves relevant chunks.",
            metadata={"source_title": "Docs", "source_url": "https://docs.example.com"},
        )
    ]
    pipeline = RagPipeline(llm=DummyLlm(), retriever=DummyRetriever(docs))

    result = pipeline.answer("What is RAG?", mode="General Research")

    assert "retrieved context" in result.answer
    assert result.citations == ["- [Docs](https://docs.example.com)"]


def test_rag_pipeline_refuses_without_context():
    pipeline = RagPipeline(llm=DummyLlm(), retriever=DummyRetriever([]))

    try:
        pipeline.answer("What is RAG?", mode="General Research")
    except InsufficientContextError as error:
        assert "enough relevant context" in str(error)
    else:
        raise AssertionError("Expected InsufficientContextError")
