import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_aws import BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

_DEFAULT_CHAT_MODEL_IDS = {
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}

_DEFAULT_EMBEDDING_MODEL_IDS = {
    "bedrock": "amazon.titan-embed-text-v2:0",
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}


def get_chat_model():
    provider = os.getenv("MODEL_PROVIDER", "bedrock")
    model_id = os.getenv("CHAT_MODEL_ID") or _DEFAULT_CHAT_MODEL_IDS[provider]
    return init_chat_model(model_id, model_provider=provider, temperature=0)


def get_embedding_model():
    provider = os.getenv("EMBEDDING_PROVIDER") or os.getenv("MODEL_PROVIDER", "bedrock")
    model_id = os.getenv("EMBEDDING_MODEL_ID") or _DEFAULT_EMBEDDING_MODEL_IDS[provider]

    if provider == "bedrock":
        return BedrockEmbeddings(model_id=model_id)
    if provider == "openai":
        return OpenAIEmbeddings(model=model_id)
    if provider == "ollama":
        return OllamaEmbeddings(model=model_id)
    raise ValueError(f"Unsupported embedding provider: {provider}")
