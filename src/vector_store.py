from __future__ import annotations

import hashlib
import math
import re

from langchain_core.documents import Document


def collection_name_for_query(query: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", query.lower()).strip("-")[:36]
    return f"research-{slug or 'query'}-{digest}"


def document_id(document: Document) -> str:
    source = document.metadata.get("source_url", "")
    index = document.metadata.get("chunk_index", 0)
    raw = f"{source}|{index}|{document.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class VectorStoreManager:
    def __init__(self, persist_directory: str, embeddings) -> None:
        self.persist_directory = persist_directory
        self.embeddings = embeddings

    def build_store(self, documents: list[Document], query: str) -> Chroma:
        from langchain_chroma import Chroma

        collection_name = collection_name_for_query(query)
        store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        if documents:
            store.add_documents(documents, ids=[document_id(document) for document in documents])
        return store

    def build_retriever(self, documents: list[Document], query: str, top_k: int = 5):
        store = self.build_store(documents, query)
        return store.as_retriever(search_kwargs={"k": top_k})


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryRetriever:
    def __init__(self, documents: list[Document], embeddings, top_k: int) -> None:
        self.documents = documents
        self.embeddings = embeddings
        self.top_k = top_k
        self.document_vectors = embeddings.embed_documents(
            [document.page_content for document in documents]
        )

    def invoke(self, query: str) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        scored = [
            (_cosine_similarity(query_vector, document_vector), document)
            for document, document_vector in zip(self.documents, self.document_vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for score, document in scored[: self.top_k] if score > 0]


class InMemoryVectorStoreManager:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def build_retriever(self, documents: list[Document], top_k: int = 5) -> InMemoryRetriever:
        return InMemoryRetriever(documents, self.embeddings, top_k)
