from __future__ import annotations

import streamlit as st

from src.config import AppConfig, MissingApiKeyError
from src.document_loader import hydrate_missing_content, prepare_documents
from src.embeddings import build_embeddings
from src.evaluation import build_retrieval_diagnostics
from src.exa_search import ExaSearchClient
from src.prompts import Mode
from src.rag_chain import InsufficientContextError, RagPipeline, build_llm
from src.text_splitter import chunk_documents
from src.utils import build_markdown_report, source_card_summary
from src.vector_store import InMemoryVectorStoreManager


st.set_page_config(
    page_title="AI Roadmap & Project Research Copilot",
    page_icon="AI",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f8fbff 0%, #ffffff 42%); }
    .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    label, [data-testid="stWidgetLabel"] {
        color: #172033 !important;
    }
    [data-testid="stSidebar"] {
        background: #171b26;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #0f1320 !important;
        border-color: #343b4f !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
    }
    [role="listbox"],
    [data-baseweb="popover"] {
        background: #ffffff !important;
    }
    [role="option"],
    [role="option"] * {
        color: #172033 !important;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #dbe4f0 !important;
    }
    .block-container { padding-top: 2.2rem; max-width: 1180px; }
    .hero {
        border: 1px solid #dbe7f5;
        border-radius: 8px;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1.1rem;
        background: linear-gradient(135deg, #ffffff 0%, #f3f8ff 100%);
        box-shadow: 0 18px 45px rgba(31, 65, 114, .08);
    }
    .app-title { font-size: 2.45rem; line-height: 1.05; font-weight: 780; margin-bottom: .45rem; color: #172033; letter-spacing: 0; }
    .app-subtitle { color: #52616f; font-size: 1.03rem; margin-bottom: 0; max-width: 790px; }
    .source-card {
        border: 1px solid #dde4ec;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: .8rem;
        background: #ffffff;
    }
    .source-meta { color: #65758b; font-size: .84rem; margin: .35rem 0; }
    .mode-help {
        border-left: 3px solid #2563eb;
        padding: .7rem .9rem;
        background: #f7faff;
        border-radius: 6px;
        color: #27364a;
        margin: .7rem 0 1rem;
    }
    .answer-shell {
        border: 1px solid #dbe7f5;
        border-radius: 8px;
        padding: .95rem 1.15rem;
        background: #ffffff;
        box-shadow: 0 14px 32px rgba(31, 65, 114, .06);
        margin-top: .35rem;
    }
    .metric-card {
        border: 1px solid #dbe7f5;
        border-radius: 8px;
        padding: .95rem 1rem;
        background: #ffffff;
        box-shadow: 0 12px 28px rgba(31, 65, 114, .06);
    }
    .metric-label { color: #64748b; font-size: .82rem; font-weight: 650; margin-bottom: .3rem; }
    .metric-value { color: #172033; font-size: 1.45rem; font-weight: 780; line-height: 1; }
    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem;
        margin-bottom: 1rem;
    }
    .workflow-card {
        border: 1px solid #dbe7f5;
        border-radius: 8px;
        padding: .9rem 1rem;
        background: #ffffff;
    }
    .workflow-title { color: #173b75; font-weight: 760; margin-bottom: .25rem; }
    .workflow-copy { color: #52616f; font-size: .92rem; line-height: 1.45; }
    @media (max-width: 800px) {
        .workflow-grid { grid-template-columns: 1fr; }
    }
    .answer-shell h3, .stMarkdown h3 {
        color: #173b75 !important;
        font-size: 1.16rem;
        margin-top: 1.15rem;
        margin-bottom: .45rem;
    }
    .answer-shell p, .answer-shell li, .source-card, .source-card div, .source-card strong {
        color: #172033 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #334155 !important;
        font-weight: 650;
    }
    div[data-testid="stButton"] button {
        border-radius: 7px;
        font-weight: 680;
    }
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"] p,
    div[data-testid="stButton"] button[kind="primary"] span {
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button[kind="secondary"],
    div[data-testid="stButton"] button[kind="secondary"] p,
    div[data-testid="stButton"] button[kind="secondary"] span {
        color: #172033 !important;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 8px;
        border-color: #ced9e6;
        background: #ffffff !important;
        color: #172033 !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #64748b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


EXAMPLES = {
    Mode.GENERAL_RESEARCH.value: "What is retrieval-augmented generation and when should I use it?",
    Mode.ROADMAP_VALIDATOR.value: "Python basics → ML → Deep Learning → LLMs. Is this roadmap complete for becoming an AI engineer?",
    Mode.PROJECT_IDEA_GENERATOR.value: "Suggest AI projects using RAG, LangChain, and a vector database.",
    Mode.LEARNING_PLAN_GENERATOR.value: "Make a 30-day plan to learn RAG and LangChain as a beginner.",
}


def render_source_cards(sources) -> None:
    for source in sources:
        score = f" · relevance {source.relevance_score:.2f}" if source.relevance_score is not None else ""
        st.markdown(
            f"""
            <div class="source-card">
                <strong>{source.title}</strong>
                <div class="source-meta">{source.source_type}{score}</div>
                <div>{source_card_summary(source.summary, source.content)}</div>
                <a href="{source.url}" target="_blank">Open source</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="app-title">AI Roadmap & Project Research Copilot</div>
            <div class="app-subtitle">
                Validate learning roadmaps, discover practical projects, and build focused study plans from current source-backed research.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="workflow-grid">
            <div class="workflow-card">
                <div class="workflow-title">Research Intake</div>
                <div class="workflow-copy">Choose a mode and submit a roadmap, topic, or project request.</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-title">Vector Retrieval</div>
                <div class="workflow-copy">Relevant source chunks are indexed and retrieved for grounded context.</div>
            </div>
            <div class="workflow-card">
                <div class="workflow-title">Evaluation View</div>
                <div class="workflow-copy">Review source mix, citation coverage, and retrieval evidence diagnostics.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Run settings")
        mode = st.selectbox("Mode", [mode.value for mode in Mode], index=0)
        num_results = st.slider("Search results", 3, 10, 5)
        top_k = st.slider("Retrieved chunks", 2, 10, 5)
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 50, 400, 150, step=25)

    st.markdown(f'<div class="mode-help">{EXAMPLES[mode]}</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Question, roadmap, or project request",
        height=180,
        max_chars=4000,
        placeholder=EXAMPLES[mode],
    )

    submitted = st.button("Research with RAG", type="primary", use_container_width=True)
    if not submitted:
        return

    if not query.strip():
        st.error("Please enter a question or roadmap.")
        return

    try:
        config = AppConfig.from_env()
    except MissingApiKeyError as exc:
        st.error("Required research and model API keys are missing.")
        st.info("Add the required API keys in your deployment secrets or local environment, then restart the app.")
        return

    try:
        with st.status("Searching current web sources...", expanded=True) as status:
            search_client = ExaSearchClient(config.exa_api_key)
            sources = search_client.search(query, num_results=num_results)
            if not sources:
                st.error("No relevant sources were found.")
                return
            st.write(f"Found {len(sources)} sources.")

            status.update(label="Preparing retrieved content...", state="running")
            hydrated_sources = hydrate_missing_content(sources)
            prepared_sources = prepare_documents(hydrated_sources, topic=query)
            if not prepared_sources:
                st.error("Could not process retrieved content.")
                return
            chunks = chunk_documents(prepared_sources, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                st.error("Could not process retrieved content.")
                return
            st.write(f"Prepared {len(chunks)} chunks.")

            status.update(label="Indexing vector chunks...", state="running")
            embeddings = build_embeddings(config.embedding_model, config.google_api_key)
            manager = InMemoryVectorStoreManager(embeddings)
            retriever = manager.build_retriever(chunks, top_k=top_k)

            status.update(label="Generating grounded answer...", state="running")
            pipeline = RagPipeline(build_llm(config.chat_model, config.google_api_key), retriever)
            result = pipeline.answer(query, mode=mode)
            status.update(label="Research complete.", state="complete")
    except InsufficientContextError as exc:
        st.warning(str(exc))
        return
    except RuntimeError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - keeps the MVP UI from crashing on provider errors.
        st.error("Answer generation failed. Please retry.")
        st.caption(str(exc))
        return

    answer_tab, sources_tab, eval_tab, export_tab = st.tabs(["Answer", "Sources", "Evaluation", "Export"])
    with answer_tab:
        st.markdown('<div class="answer-shell">', unsafe_allow_html=True)
        st.markdown(result.answer)
        st.markdown("</div>", unsafe_allow_html=True)
        if result.citations:
            st.subheader("Citations")
            st.markdown("\n".join(result.citations))

    with sources_tab:
        render_source_cards(sources)

    with eval_tab:
        diagnostics = build_retrieval_diagnostics(sources, chunks, result.documents, result.citations)
        st.subheader("Retrieval Diagnostics")
        metric_items = list(diagnostics.metrics.items())
        for row_start in range(0, len(metric_items), 3):
            cols = st.columns(3)
            for col, (label_text, value) in zip(cols, metric_items[row_start : row_start + 3]):
                col.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label_text}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("### Evidence Matrix")
        st.dataframe(diagnostics.matrix, hide_index=True, use_container_width=True)
        st.markdown("### Source Mix")
        st.dataframe(diagnostics.source_type_rows, hide_index=True, use_container_width=True)
        st.caption("These are retrieval diagnostics for the RAG pipeline, not labeled ML benchmark scores.")

    with export_tab:
        report = build_markdown_report(query, mode, result.answer, result.citations)
        st.download_button(
            "Download Markdown report",
            data=report,
            file_name="ai-roadmap-research-report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.markdown(report)


if __name__ == "__main__":
    main()
