import pytest

from src.config import AppConfig, MissingApiKeyError


def test_config_loads_required_keys_from_environment(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "exa-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")

    config = AppConfig.from_env()

    assert config.exa_api_key == "exa-test"
    assert config.google_api_key == "google-test"
    assert config.chat_model == "gemini-3-flash-preview"
    assert config.embedding_model == "gemini-embedding-001"
    assert config.chroma_path == "data/chroma_db"


def test_config_reports_missing_required_keys(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(MissingApiKeyError) as error:
        AppConfig.from_env(load_dotenv_file=False)

    assert "EXA_API_KEY" in str(error.value)
    assert "GOOGLE_API_KEY" in str(error.value)
