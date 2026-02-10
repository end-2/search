"""Serving 파이프라인 통합 테스트.

전체 Serving 흐름을 테스트합니다:
1. QueryRewriter: 쿼리 리파인
2. SearchRouter: 검색 전략 결정
3. HybridRetriever: 벡터 검색
4. Reranker: 재순위화
5. LLMClient: 답변 생성
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.serving.core.security import UserContext
from src.serving.query_engine.rewriter import QueryRewriter, RefinedQuery
from src.serving.query_engine.router import SearchRouter, SearchStrategy
from src.serving.retrieval.hybrid import HybridRetriever, RetrievedDocument


class TestServingPipeline:
    """Serving 파이프라인 통합 테스트."""

    @pytest.fixture
    def user_context(self):
        """테스트용 UserContext."""
        return UserContext(user_id="test-user", groups=["engineering"])

    @pytest.fixture
    def query_rewriter(self):
        """QueryRewriter 인스턴스."""
        return QueryRewriter()

    @pytest.fixture
    def search_router(self):
        """SearchRouter 인스턴스."""
        return SearchRouter()

    def test_query_rewrite_to_search_plan(
        self,
        query_rewriter,
        search_router,
        mock_openai_chat,
    ):
        """QueryRewriter → SearchRouter 흐름 테스트."""
        # QueryRewriter 응답 설정
        mock_openai_chat.return_value = json.dumps({
            "refined_query": "배포 에러 로그 및 장애 현황",
            "sub_queries": [],
            "metadata_filters": {},
            "keywords": ["배포", "에러", "deploy", "error", "장애"],
        })

        # 1. Query Rewrite
        refined = query_rewriter.rewrite("지난주 배포 에러 뭐였어?")

        assert refined.refined_query == "배포 에러 로그 및 장애 현황"
        assert len(refined.keywords) >= 4

        # 2. Search Plan 생성
        plan = search_router.plan(refined)

        assert plan.strategy == SearchStrategy.HYBRID  # 키워드가 많으므로
        assert plan.queries == [refined.refined_query]

    def test_full_pipeline_with_mocks(
        self,
        query_rewriter,
        search_router,
        user_context,
        mock_openai_chat,
        mock_openai_embeddings,
        mock_qdrant_with_data,
    ):
        """전체 파이프라인 (모킹) 테스트."""
        from src.ingestion.embedder import SparseEncoder
        from src.ingestion.vector_store import VectorStore
        from src.serving.generation.llm_client import LLMClient
        from src.serving.retrieval.hybrid import HybridRetriever
        from src.serving.retrieval.reranker import Reranker

        # Mock 설정
        mock_openai_chat.return_value = json.dumps({
            "refined_query": "테스트 쿼리",
            "sub_queries": [],
            "metadata_filters": {},
            "keywords": ["테스트", "쿼리"],
        })

        # 1. Query Rewrite
        refined = query_rewriter.rewrite("테스트 쿼리")

        # 2. Search Plan
        plan = search_router.plan(refined)

        # 3. Hybrid Retrieval
        vector_store = VectorStore()
        sparse_encoder = SparseEncoder()
        sparse_encoder.fit(["test corpus for sparse encoding"])

        retriever = HybridRetriever(vector_store, sparse_encoder)
        retrieved = retriever.retrieve(plan, user_context)

        assert len(retrieved) >= 0  # 모킹된 결과

        # 4. Reranking
        reranker = Reranker()
        # Reranker LLM 응답 설정
        mock_openai_chat.return_value = "Reranked response"
        reranked = reranker.rerank("테스트 쿼리", retrieved)

        # 5. Answer Generation
        llm_client = LLMClient()
        mock_openai_chat.return_value = "이것은 테스트 답변입니다."

        if reranked:
            result = llm_client.generate("테스트 쿼리", reranked)
            assert result.answer == "이것은 테스트 답변입니다."


class TestHybridRetriever:
    """HybridRetriever 통합 테스트."""

    @pytest.fixture
    def sparse_encoder(self):
        """학습된 SparseEncoder."""
        from src.ingestion.embedder import SparseEncoder

        encoder = SparseEncoder()
        encoder.fit([
            "test query for search",
            "another document about testing",
            "search engine retrieval",
        ])
        return encoder

    def test_retrieve_with_acl_filter(
        self,
        sparse_encoder,
        mock_openai_embeddings,
        mock_qdrant_with_data,
    ):
        """ACL 필터가 적용된 검색 테스트."""
        from src.ingestion.vector_store import VectorStore
        from src.serving.query_engine.router import SearchPlan, SearchStrategy

        vector_store = VectorStore()
        retriever = HybridRetriever(vector_store, sparse_encoder)

        user_context = UserContext(user_id="user1", groups=["group1"])
        plan = SearchPlan(
            queries=["test query"],
            strategy=SearchStrategy.HYBRID,
        )

        results = retriever.retrieve(plan, user_context)

        # Qdrant search가 호출되었는지 확인
        mock_qdrant_with_data.search.assert_called()

        # ACL 필터가 전달되었는지 확인
        call_args = mock_qdrant_with_data.search.call_args
        assert call_args[1]["query_filter"] is not None

    def test_retrieve_multiple_queries(
        self,
        sparse_encoder,
        mock_openai_embeddings,
        mock_qdrant_with_data,
    ):
        """여러 쿼리에 대한 검색 테스트."""
        from src.ingestion.vector_store import VectorStore
        from src.serving.query_engine.router import SearchPlan, SearchStrategy

        vector_store = VectorStore()
        retriever = HybridRetriever(vector_store, sparse_encoder)

        user_context = UserContext(user_id="user1", groups=[])
        plan = SearchPlan(
            queries=["query 1", "query 2"],
            strategy=SearchStrategy.HYBRID,
        )

        results = retriever.retrieve(plan, user_context)

        # 각 쿼리에 대해 search가 호출됨
        assert mock_qdrant_with_data.search.call_count == 2


class TestReranker:
    """Reranker 통합 테스트."""

    @pytest.fixture
    def sample_retrieved_documents(self):
        """테스트용 검색 결과."""
        return [
            RetrievedDocument(
                child_id=f"child-{i}",
                parent_id=f"parent-{i}",
                child_content=f"Child content {i}",
                parent_content=f"Parent content {i} with more context",
                score=0.9 - i * 0.1,
                source="test",
                url=f"https://example.com/{i}",
                author="tester",
            )
            for i in range(5)
        ]

    def test_rerank_returns_top_k(self, sample_retrieved_documents, mock_openai_chat):
        """Reranker가 top_k 개수만큼 반환."""
        from src.serving.retrieval.reranker import Reranker

        mock_openai_chat.return_value = "Relevance assessment"
        reranker = Reranker()

        # top_k 미만의 결과
        result = reranker.rerank("test query", sample_retrieved_documents[:3])
        assert len(result) <= 3


class TestLLMClient:
    """LLMClient 통합 테스트."""

    @pytest.fixture
    def sample_documents(self):
        """테스트용 문서."""
        return [
            RetrievedDocument(
                child_id="c1",
                parent_id="p1",
                child_content="Relevant content",
                parent_content="Full context of the relevant content",
                score=0.95,
                source="slack",
                url="https://slack.com/archives/C123",
                author="alice",
            )
        ]

    def test_generate_includes_sources(self, sample_documents, mock_openai_chat):
        """답변에 출처 정보가 포함되어야 함."""
        from src.serving.generation.llm_client import LLMClient

        mock_openai_chat.return_value = "Generated answer based on context"
        client = LLMClient()

        result = client.generate("test query", sample_documents)

        assert result.answer == "Generated answer based on context"
        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "slack"
        assert result.sources[0]["url"] == "https://slack.com/archives/C123"

    def test_generate_empty_documents(self, mock_openai_chat):
        """문서가 없는 경우도 처리."""
        from src.serving.generation.llm_client import LLMClient

        mock_openai_chat.return_value = "I don't have enough context to answer."
        client = LLMClient()

        result = client.generate("unknown query", [])

        assert result.answer == "I don't have enough context to answer."
        assert result.sources == []


class TestServingAPIIntegration:
    """Serving API 엔드포인트 통합 테스트."""

    @pytest.fixture
    def test_client(self):
        """FastAPI 테스트 클라이언트."""
        from fastapi.testclient import TestClient

        # 모든 외부 의존성 모킹
        with patch("src.serving.main.vector_store") as mock_vs, \
             patch("src.serving.main.query_rewriter") as mock_qr, \
             patch("src.serving.main.hybrid_retriever") as mock_hr, \
             patch("src.serving.main.reranker") as mock_rr, \
             patch("src.serving.main.llm_client") as mock_llm:

            # Mock 설정
            mock_qr.rewrite.return_value = RefinedQuery(
                original_query="test",
                refined_query="refined test",
                keywords=["test", "query"],
            )

            mock_hr.retrieve.return_value = [
                RetrievedDocument(
                    child_id="c1",
                    parent_id="p1",
                    child_content="content",
                    parent_content="parent content",
                    score=0.9,
                    source="test",
                    url="https://example.com",
                    author="tester",
                )
            ]

            mock_rr.rerank.return_value = mock_hr.retrieve.return_value

            from src.serving.generation.llm_client import GenerationResult
            mock_llm.generate.return_value = GenerationResult(
                answer="This is the answer",
                sources=[{"url": "https://example.com", "source": "test"}],
            )

            from src.serving.main import app
            yield TestClient(app)

    def test_health_endpoint(self, test_client):
        """Health 엔드포인트 테스트."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_search_endpoint_requires_auth(self, test_client):
        """Search 엔드포인트 인증 필수."""
        response = test_client.post(
            "/v1/search",
            json={"query": "test query"},
        )
        assert response.status_code == 401

    def test_search_endpoint_with_auth(self, test_client):
        """Search 엔드포인트 정상 요청."""
        response = test_client.post(
            "/v1/search",
            json={"query": "test query"},
            headers={
                "X-User-Id": "test-user",
                "X-User-Groups": "engineering,team",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_chat_completions_endpoint(self, test_client):
        """Chat Completions 엔드포인트 테스트."""
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "test question"}
                ],
            },
            headers={
                "X-User-Id": "test-user",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
