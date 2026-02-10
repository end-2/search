import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.common.openai_utils import chat_completion

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are a query optimization assistant for a RAG search system.
Your job is to take a user's natural language question and produce a structured search plan.

You must return a JSON object with the following fields:
- "refined_query": The main search query, optimized for semantic search. Expand abbreviations and domain-specific terms.
- "sub_queries": A list of sub-queries if the original question is complex and can be decomposed. Otherwise, an empty list.
- "metadata_filters": An object with optional fields like "date_from", "date_to", "author", "source" extracted from the question. Only include fields that are explicitly mentioned.
- "keywords": A list of important keywords for keyword-based search.

Examples:
User: "지난주 배포 에러 뭐였지?"
Output:
{
  "refined_query": "최근 배포 과정에서 발생한 에러 및 장애 내역",
  "sub_queries": [],
  "metadata_filters": {},
  "keywords": ["배포", "에러", "deploy", "error", "fail", "장애"]
}

User: "Alice가 작성한 HR 정책 변경사항과 관련 Jira 티켓 알려줘"
Output:
{
  "refined_query": "HR 인사팀 정책 변경사항 및 관련 Jira 이슈",
  "sub_queries": ["HR 인사팀 정책 변경사항", "HR 정책 관련 Jira 티켓"],
  "metadata_filters": {"author": "Alice"},
  "keywords": ["HR", "인사팀", "정책", "변경", "Jira"]
}

Return ONLY the JSON object, no additional text."""


@dataclass
class RefinedQuery:
    """쿼리 리파인 결과."""

    original_query: str
    refined_query: str
    sub_queries: List[str] = field(default_factory=list)
    metadata_filters: Dict = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)


class QueryRewriter:
    """
    LLM 기반 Query Rewriter.
    사용자의 자연어 질문을 검색에 최적화된 형태로 변환합니다.
    - Expansion: 동의어/약어 확장
    - Decomposition: 복합 질문 분해
    - Metadata Extraction: 날짜, 작성자 등 필터 조건 추출
    """

    def rewrite(self, query: str) -> RefinedQuery:
        """사용자 쿼리를 분석하고 검색에 최적화된 형태로 변환합니다."""
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        response_text = chat_completion(messages, temperature=0.0)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning(f"Query rewrite JSON 파싱 실패, 원본 쿼리 사용: {response_text}")
            return RefinedQuery(
                original_query=query,
                refined_query=query,
                keywords=query.split(),
            )

        return RefinedQuery(
            original_query=query,
            refined_query=result.get("refined_query", query),
            sub_queries=result.get("sub_queries", []),
            metadata_filters=result.get("metadata_filters", {}),
            keywords=result.get("keywords", []),
        )
