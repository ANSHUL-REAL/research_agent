from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class MissingApiKeyError(RuntimeError):
    """Raised when required API keys are not available."""


@dataclass(frozen=True)
class AppConfig:
    exa_api_key: str
    google_api_key: str
    chat_model: str = "gemini-3-flash-preview"
    embedding_model: str = "gemini-embedding-001"
    chroma_path: str = "data/chroma_db"

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "AppConfig":
        if load_dotenv_file:
            load_dotenv()
        exa_api_key = os.getenv("EXA_API_KEY", "").strip()
        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        missing = [
            name
            for name, value in (
                ("EXA_API_KEY", exa_api_key),
                ("GOOGLE_API_KEY", google_api_key),
            )
            if not value
        ]
        if missing:
            raise MissingApiKeyError(
                "Missing required API key(s): "
                + ", ".join(missing)
                + ". Add them to your deployment secrets or local environment."
            )

        return cls(
            exa_api_key=exa_api_key,
            google_api_key=google_api_key,
            chat_model=os.getenv("GEMINI_CHAT_MODEL", cls.chat_model).strip() or cls.chat_model,
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", cls.embedding_model).strip()
            or cls.embedding_model,
            chroma_path=os.getenv("CHROMA_PATH", cls.chroma_path).strip() or cls.chroma_path,
        )
