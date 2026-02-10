import logging
from typing import List

from openai import OpenAI

from src.common.config import get_settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """텍스트 리스트를 Dense Vector로 변환합니다."""
    settings = get_settings()
    client = get_openai_client()
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def chat_completion(
    messages: List[dict],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Chat Completion API를 호출하여 응답 텍스트를 반환합니다."""
    settings = get_settings()
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model or settings.openai_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
