from src.exa_search import ExaSearchClient, infer_source_type, normalize_exa_result


class DummyResult:
    title = "LangChain docs"
    url = "https://python.langchain.com/docs/tutorials/rag/"
    text = "Build a RAG app with LangChain."
    highlights = ["RAG tutorial"]
    score = 0.91


class DummyResponse:
    results = [DummyResult()]


class DummyExa:
    def search_and_contents(self, query, type, num_results, text, highlights):
        assert query == "rag langchain"
        assert type == "auto"
        assert num_results == 1
        assert text is True
        assert highlights is True
        return DummyResponse()


def test_infer_source_type_prioritizes_docs_and_github():
    assert infer_source_type("https://docs.python.org/3/") == "Official docs"
    assert infer_source_type("https://github.com/langchain-ai/langchain") == "GitHub"
    assert infer_source_type("https://arxiv.org/abs/2401.00001") == "Research paper"


def test_normalize_exa_result_extracts_content_and_metadata():
    source = normalize_exa_result(DummyResult())

    assert source.title == "LangChain docs"
    assert source.url == "https://python.langchain.com/docs/tutorials/rag/"
    assert source.summary == "RAG tutorial"
    assert source.content == "Build a RAG app with LangChain."
    assert source.relevance_score == 0.91
    assert source.source_type == "Official docs"


def test_search_client_returns_normalized_sources():
    client = ExaSearchClient(api_key="test", exa_client=DummyExa())

    sources = client.search("rag langchain", num_results=1)

    assert len(sources) == 1
    assert sources[0].title == "LangChain docs"
