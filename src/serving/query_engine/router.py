import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.serving.query_engine.rewriter import RefinedQuery

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    HYBRID = "hybrid"
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"


@dataclass
class SearchPlan:
    """검색 실행 계획."""

    queries: List[str]
    strategy: SearchStrategy = SearchStrategy.HYBRID
    source_filter: Optional[str] = None
    metadata_filters: dict = field(default_factory=dict)


class SearchRouter:
    """
    검색 전략을 결정하는 라우터.
    질문의 특성에 따라 검색 방식과 데이터 소스를 결정합니다.
    """

    SOURCE_KEYWORDS = {
        "slack": ["슬랙", "slack", "채널", "channel", "메시지", "message", "대화"],
        "jira": ["지라", "jira", "티켓", "ticket", "이슈", "issue", "스프린트", "sprint"],
    }

    def plan(self, refined_query: RefinedQuery) -> SearchPlan:
        """RefinedQuery를 기반으로 검색 계획을 수립합니다."""
        # 하위 질문이 있으면 각각 검색, 없으면 refined_query 하나만 검색
        queries = refined_query.sub_queries if refined_query.sub_queries else [refined_query.refined_query]

        # 메타데이터 필터에서 소스가 명시된 경우
        source_filter = refined_query.metadata_filters.get("source")

        # 키워드에서 소스 힌트 추출
        if not source_filter:
            source_filter = self._detect_source(refined_query)

        # 키워드가 충분하면 hybrid, 부족하면 dense 우선
        strategy = SearchStrategy.HYBRID
        if len(refined_query.keywords) < 2:
            strategy = SearchStrategy.DENSE_ONLY

        return SearchPlan(
            queries=queries,
            strategy=strategy,
            source_filter=source_filter,
            metadata_filters=refined_query.metadata_filters,
        )

    def _detect_source(self, refined_query: RefinedQuery) -> Optional[str]:
        """질문 내용에서 데이터 소스 힌트를 탐지합니다."""
        all_text = (
            refined_query.original_query.lower()
            + " "
            + " ".join(refined_query.keywords).lower()
        )
        for source, keywords in self.SOURCE_KEYWORDS.items():
            if any(kw in all_text for kw in keywords):
                return source
        return None
