import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.common.config import get_settings
from src.common.openai_utils import create_embeddings
from src.ingestion.embedder import SparseEncoder
from src.ingestion.vector_store import VectorStore
from src.serving.core.security import UserContext
from src.serving.query_engine.router import SearchPlan, SearchStrategy

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """검색된 문서 (Parent Chunk 포함)."""

    child_id: str
    parent_id: str
    child_content: str
    parent_content: str
    score: float
    source: str
    url: str
    author: str
    metadata: Dict = field(default_factory=dict)


class HybridRetriever:
    """
    Secure Hybrid Retriever.
    - Dense + Sparse 결합 검색
    - ACL 기반 보안 필터링
    - Small-to-Big: Child로 검색 → Parent로 문맥 확보
    """

    def __init__(self, vector_store: VectorStore, sparse_encoder: SparseEncoder):
        self._store = vector_store
        self._sparse_encoder = sparse_encoder
        self._settings = get_settings()

    def retrieve(
        self,
        search_plan: SearchPlan,
        user_context: UserContext,
    ) -> List[RetrievedDocument]:
        """
        검색 계획에 따라 문서를 검색하고 Parent Chunk로 확장합니다.

        1. 각 쿼리에 대해 Child Chunk 벡터 검색 수행
        2. ACL 필터 적용
        3. 검색된 Child의 Parent Chunk를 조회하여 문맥 확보
        """
        all_results: Dict[str, RetrievedDocument] = {}

        for query in search_plan.queries:
            child_results = self._search_query(
                query=query,
                strategy=search_plan.strategy,
                user_context=user_context,
                source_filter=search_plan.source_filter,
            )

            for result in child_results:
                # parent_id 기준으로 중복 제거 (가장 높은 score 유지)
                parent_id = result.parent_id
                if parent_id not in all_results or result.score > all_results[parent_id].score:
                    all_results[parent_id] = result

        # 점수순 정렬
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)

        top_k = self._settings.retrieval_top_k
        return sorted_results[:top_k]

    def _search_query(
        self,
        query: str,
        strategy: SearchStrategy,
        user_context: UserContext,
        source_filter: Optional[str] = None,
    ) -> List[RetrievedDocument]:
        """단일 쿼리에 대한 검색을 수행합니다."""
        # Dense Vector 생성
        dense_vector = create_embeddings([query])[0]

        # Sparse Vector 생성
        sparse_vector = None
        if strategy in (SearchStrategy.HYBRID, SearchStrategy.SPARSE_ONLY):
            sparse_vector = self._sparse_encoder.encode(query)

        # Qdrant 검색 (ACL 필터 포함)
        child_hits = self._store.search_children(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            user_id=user_context.user_id,
            user_groups=user_context.groups,
            top_k=self._settings.retrieval_top_k,
        )

        # Child → Parent 확장 (Small-to-Big Retrieval)
        results: List[RetrievedDocument] = []
        for hit in child_hits:
            payload = hit["payload"]
            parent_id = payload.get("parent_id", "")

            # 소스 필터 적용
            if source_filter and payload.get("source") != source_filter:
                continue

            # Parent Chunk 조회
            parent_content = ""
            if parent_id:
                parent = self._store.get_parent_by_id(parent_id)
                if parent:
                    parent_content = parent.get("content", "")

            results.append(
                RetrievedDocument(
                    child_id=str(hit["id"]),
                    parent_id=parent_id,
                    child_content=payload.get("content_text", ""),
                    parent_content=parent_content or payload.get("content_text", ""),
                    score=hit["score"],
                    source=payload.get("source", ""),
                    url=payload.get("url", ""),
                    author=payload.get("author", ""),
                    metadata=payload.get("metadata", {}),
                )
            )

        return results
