import os

from langchain.chat_models import init_chat_model
from langchain_aws import BedrockEmbeddings

_DEFAULT_CHAT_MODEL_IDS = {
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o-mini",
}


def get_chat_model():
    provider = os.getenv("MODEL_PROVIDER", "bedrock")
    model_id = os.getenv("CHAT_MODEL_ID") or _DEFAULT_CHAT_MODEL_IDS[provider]
    return init_chat_model(model_id, model_provider=provider, temperature=0)


def get_embedding_model():
    model_id = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    return BedrockEmbeddings(model_id=model_id)
