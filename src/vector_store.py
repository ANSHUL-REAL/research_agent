from __future__ import annotations

import hashlib
import re

from langchain_chroma import Chroma
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
