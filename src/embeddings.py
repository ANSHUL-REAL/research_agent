from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_embeddings(model: str, google_api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=google_api_key)
