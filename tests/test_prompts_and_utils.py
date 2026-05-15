import pytest

from src.prompts import Mode, get_prompt_template
from src.utils import build_markdown_report, format_citations


@pytest.mark.parametrize(
    "mode,required_text",
    [
        (Mode.GENERAL_RESEARCH, "AI research assistant"),
        (Mode.ROADMAP_VALIDATOR, "career roadmap evaluator"),
        (Mode.PROJECT_IDEA_GENERATOR, "AI project mentor"),
        (Mode.LEARNING_PLAN_GENERATOR, "AI learning planner"),
    ],
)
def test_each_mode_has_prompt_template(mode, required_text):
    prompt = get_prompt_template(mode)

    assert required_text in prompt.template
    assert "context" in prompt.input_variables
    assert "question" in prompt.input_variables


def test_project_prompt_does_not_frame_projects_as_resume_output():
    prompt = get_prompt_template(Mode.PROJECT_IDEA_GENERATOR)

    assert "resume" not in prompt.template.lower()
    assert "portfolio" not in prompt.template.lower()


def test_format_citations_deduplicates_sources():
    docs = [
        {"source_title": "Docs", "source_url": "https://docs.example.com"},
        {"source_title": "Docs", "source_url": "https://docs.example.com"},
        {"source_title": "Blog", "source_url": "https://blog.example.com"},
    ]

    citations = format_citations(docs)

    assert citations == [
        "- [Docs](https://docs.example.com)",
        "- [Blog](https://blog.example.com)",
    ]


def test_build_markdown_report_includes_sources():
    report = build_markdown_report(
        query="What is RAG?",
        mode="General Research",
        answer="RAG combines retrieval and generation.",
        citations=["- [Docs](https://docs.example.com)"],
    )

    assert "# AI Roadmap & Project Research Copilot Report" in report
    assert "What is RAG?" in report
    assert "- [Docs](https://docs.example.com)" in report
