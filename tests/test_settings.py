from unittest.mock import patch

from config.settings import get_chat_model, get_embedding_model


def test_get_chat_model_defaults_to_bedrock(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("CHAT_MODEL_ID", raising=False)
    with patch("config.settings.init_chat_model") as mock_init:
        get_chat_model()
        mock_init.assert_called_once_with(
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            model_provider="bedrock",
            temperature=0,
        )


def test_get_chat_model_respects_provider_override(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.delenv("CHAT_MODEL_ID", raising=False)
    with patch("config.settings.init_chat_model") as mock_init:
        get_chat_model()
        mock_init.assert_called_once_with(
            "claude-3-5-sonnet-20241022",
            model_provider="anthropic",
            temperature=0,
        )


def test_get_embedding_model_defaults_to_bedrock(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
    with patch("config.settings.BedrockEmbeddings") as mock_embeddings:
        get_embedding_model()
        mock_embeddings.assert_called_once_with(model_id="amazon.titan-embed-text-v2:0")


def test_get_embedding_model_follows_model_provider_by_default(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
    with patch("config.settings.OllamaEmbeddings") as mock_embeddings:
        get_embedding_model()
        mock_embeddings.assert_called_once_with(model="nomic-embed-text")


def test_get_embedding_model_respects_explicit_embedding_provider_override(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
    with patch("config.settings.BedrockEmbeddings") as mock_embeddings:
        get_embedding_model()
        mock_embeddings.assert_called_once_with(model_id="amazon.titan-embed-text-v2:0")


def test_get_embedding_model_openai_provider(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
    with patch("config.settings.OpenAIEmbeddings") as mock_embeddings:
        get_embedding_model()
        mock_embeddings.assert_called_once_with(model="text-embedding-3-small")
