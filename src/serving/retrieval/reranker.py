import logging
from typing import List

from src.common.config import get_settings
from src.common.openai_utils import chat_completion
from src.serving.retrieval.hybrid import RetrievedDocument

logger = logging.getLogger(__name__)

RERANK_SYSTEM_PROMPT = """You are a document relevance scorer.
Given a user query and a document, rate the relevance of the document to the query on a scale of 0 to 10.
Return ONLY a single integer number (0-10), nothing else.

Scoring guide:
- 0: Completely irrelevant
- 1-3: Slightly related but not useful
- 4-6: Somewhat relevant, contains partial information
- 7-9: Highly relevant, directly answers the query
- 10: Perfect match, completely answers the query"""


class Reranker:
    """
    LLM 기반 Reranker.
    1차 검색된 문서를 재순위화하여 가장 관련성 높은 Top-K를 추출합니다.
    Cross-Encoder 대신 LLM을 사용하여 별도 모델 의존성을 줄입니다.
    """

    def __init__(self):
        self._settings = get_settings()

    def rerank(
        self, query: str, documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        문서를 쿼리와의 관련성 기준으로 재순위화합니다.

        Args:
            query: 사용자 질문
            documents: 1차 검색된 문서 리스트
        Returns:
            재순위화된 Top-K 문서 리스트
        """
        top_k = self._settings.rerank_top_k

        if len(documents) <= top_k:
            return documents

        scored_docs = []
        for doc in documents:
            score = self._score_document(query, doc)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored_docs[:top_k]]

        logger.info(
            f"Reranking 완료: {len(documents)}개 → Top-{top_k} 추출"
        )
        return reranked

    def _score_document(self, query: str, doc: RetrievedDocument) -> float:
        """LLM을 사용하여 쿼리-문서 관련성 점수를 산출합니다."""
        # Parent content를 사용하여 전체 문맥 기반 판단
        content = doc.parent_content[:1500]

        messages = [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocument:\n{content}",
            },
        ]

        try:
            response = chat_completion(messages, temperature=0.0, max_tokens=5)
            score = float(response.strip())
            return min(max(score, 0.0), 10.0)
        except (ValueError, TypeError):
            logger.warning(f"Reranking 점수 파싱 실패, 기본 점수(5.0) 사용")
            return 5.0
