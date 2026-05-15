from langchain_core.documents import Document

from src.vector_store import (
    InMemoryVectorStoreManager,
    KeywordRetriever,
    collection_name_for_query,
    document_id,
)


class TinyEmbeddings:
    def embed_documents(self, texts):
        return [[float(text.count("rag")), float(text.count("python"))] for text in texts]

    def embed_query(self, text):
        return [float(text.count("rag")), float(text.count("python"))]


class QuotaEmbeddings:
    def embed_documents(self, texts):
        raise RuntimeError("429 RESOURCE_EXHAUSTED embed_content_free_tier_requests")


def test_collection_name_for_query_is_stable_and_safe():
    assert collection_name_for_query("What is RAG?") == collection_name_for_query("What is RAG?")
    assert collection_name_for_query("What is RAG?").startswith("research-what-is-rag-")


def test_document_id_is_stable():
    document = Document(page_content="rag", metadata={"source_url": "https://example.com", "chunk_index": 1})

    assert document_id(document) == document_id(document)


def test_in_memory_retriever_returns_most_similar_documents():
    documents = [
        Document(page_content="rag rag vector", metadata={"source_url": "https://rag.example.com"}),
        Document(page_content="python basics", metadata={"source_url": "https://python.example.com"}),
    ]
    retriever = InMemoryVectorStoreManager(TinyEmbeddings()).build_retriever(documents, top_k=1)

    results = retriever.invoke("rag")

    assert len(results) == 1
    assert results[0].metadata["source_url"] == "https://rag.example.com"


def test_keyword_retriever_returns_token_overlap_matches():
    documents = [
        Document(page_content="vector database retrieval augmented generation", metadata={"source_url": "https://rag.example.com"}),
        Document(page_content="frontend css streamlit controls", metadata={"source_url": "https://ui.example.com"}),
    ]
    retriever = KeywordRetriever(documents, top_k=1)

    results = retriever.invoke("rag vector retrieval")

    assert len(results) == 1
    assert results[0].metadata["source_url"] == "https://rag.example.com"


def test_manager_falls_back_to_keyword_retrieval_on_embedding_quota_error():
    documents = [
        Document(page_content="rag vector retrieval", metadata={"source_url": "https://rag.example.com"}),
    ]
    manager = InMemoryVectorStoreManager(QuotaEmbeddings())

    retriever = manager.build_retriever(documents, top_k=1)

    assert isinstance(retriever, KeywordRetriever)
    assert manager.last_retrieval_mode == "keyword"
