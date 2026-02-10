import logging
from dataclasses import dataclass, field
from typing import List

from src.common.openai_utils import chat_completion
from src.serving.generation.prompt import PromptBuilder
from src.serving.retrieval.hybrid import RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """답변 생성 결과."""

    answer: str
    sources: List[dict] = field(default_factory=list)


class LLMClient:
    """
    OpenAI Chat Completion API를 사용하여 최종 답변을 생성합니다.
    검색된 Context와 사용자 질문을 결합하여 출처가 포함된 답변을 반환합니다.
    """

    def __init__(self):
        self._prompt_builder = PromptBuilder()

    def generate(
        self, query: str, documents: List[RetrievedDocument]
    ) -> GenerationResult:
        """
        검색된 문서를 기반으로 답변을 생성합니다.

        Args:
            query: 사용자 질문
            documents: Reranking 완료된 Top-K 문서
        Returns:
            GenerationResult (답변 텍스트 + 출처 목록)
        """
        messages = self._prompt_builder.build(query, documents)
        answer = chat_completion(messages, temperature=0.2)

        sources = [
            {
                "title": f"[{doc.source.upper()}] {doc.url.split('/')[-1]}" if doc.url else doc.source,
                "url": doc.url,
                "source": doc.source,
                "author": doc.author,
            }
            for doc in documents
            if doc.url
        ]

        logger.info(f"답변 생성 완료 (출처 {len(sources)}개)")
        return GenerationResult(answer=answer, sources=sources)
