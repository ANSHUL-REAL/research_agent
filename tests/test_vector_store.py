from langchain_core.documents import Document

from src.vector_store import InMemoryVectorStoreManager, collection_name_for_query, document_id


class TinyEmbeddings:
    def embed_documents(self, texts):
        return [[float(text.count("rag")), float(text.count("python"))] for text in texts]

    def embed_query(self, text):
        return [float(text.count("rag")), float(text.count("python"))]


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
