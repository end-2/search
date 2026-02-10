"""SearchRouter 모듈 테스트."""

import pytest

from src.serving.query_engine.rewriter import RefinedQuery
from src.serving.query_engine.router import SearchPlan, SearchRouter, SearchStrategy


class TestSearchRouter:
    """SearchRouter 테스트."""

    @pytest.fixture
    def router(self):
        """SearchRouter 인스턴스."""
        return SearchRouter()

    @pytest.fixture
    def basic_refined_query(self):
        """기본 RefinedQuery."""
        return RefinedQuery(
            original_query="배포 에러 알려줘",
            refined_query="배포 과정에서 발생한 에러",
            sub_queries=[],
            metadata_filters={},
            keywords=["배포", "에러", "deploy", "error"],
        )

    def test_plan_returns_search_plan(self, router, basic_refined_query):
        """plan이 SearchPlan 객체를 반환해야 함."""
        plan = router.plan(basic_refined_query)
        assert isinstance(plan, SearchPlan)

    def test_plan_uses_refined_query_when_no_sub_queries(self, router, basic_refined_query):
        """sub_queries가 없으면 refined_query를 사용해야 함."""
        plan = router.plan(basic_refined_query)
        assert plan.queries == ["배포 과정에서 발생한 에러"]

    def test_plan_uses_sub_queries_when_present(self, router):
        """sub_queries가 있으면 해당 쿼리들을 사용해야 함."""
        refined_query = RefinedQuery(
            original_query="HR 정책과 Jira 티켓",
            refined_query="HR 정책 및 관련 티켓",
            sub_queries=["HR 정책 변경사항", "HR 관련 Jira 티켓"],
            metadata_filters={},
            keywords=["HR", "정책", "Jira"],
        )
        plan = router.plan(refined_query)
        assert plan.queries == ["HR 정책 변경사항", "HR 관련 Jira 티켓"]

    def test_plan_hybrid_strategy_with_keywords(self, router, basic_refined_query):
        """키워드가 2개 이상이면 hybrid 전략 사용."""
        plan = router.plan(basic_refined_query)
        assert plan.strategy == SearchStrategy.HYBRID

    def test_plan_dense_only_with_few_keywords(self, router):
        """키워드가 2개 미만이면 dense_only 전략 사용."""
        refined_query = RefinedQuery(
            original_query="뭔가 찾아줘",
            refined_query="검색할 내용",
            sub_queries=[],
            metadata_filters={},
            keywords=["검색"],  # 1개 키워드
        )
        plan = router.plan(refined_query)
        assert plan.strategy == SearchStrategy.DENSE_ONLY

    def test_plan_source_filter_from_metadata(self, router):
        """metadata_filters에 source가 있으면 source_filter 설정."""
        refined_query = RefinedQuery(
            original_query="슬랙에서 찾아줘",
            refined_query="슬랙 메시지 검색",
            sub_queries=[],
            metadata_filters={"source": "slack"},
            keywords=["슬랙", "메시지"],
        )
        plan = router.plan(refined_query)
        assert plan.source_filter == "slack"


class TestSearchRouterSourceDetection:
    """SearchRouter._detect_source 테스트."""

    @pytest.fixture
    def router(self):
        return SearchRouter()

    def test_detect_slack_from_original_query(self, router):
        """original_query에서 slack 키워드 감지."""
        refined_query = RefinedQuery(
            original_query="슬랙에서 대화 찾아줘",
            refined_query="대화 검색",
            keywords=[],
        )
        source = router._detect_source(refined_query)
        assert source == "slack"

    def test_detect_slack_from_keywords(self, router):
        """keywords에서 slack 키워드 감지."""
        refined_query = RefinedQuery(
            original_query="메시지 찾아줘",
            refined_query="메시지 검색",
            keywords=["channel", "message"],
        )
        source = router._detect_source(refined_query)
        assert source == "slack"

    def test_detect_jira_from_original_query(self, router):
        """original_query에서 jira 키워드 감지."""
        refined_query = RefinedQuery(
            original_query="지라 티켓 보여줘",
            refined_query="티켓 검색",
            keywords=[],
        )
        source = router._detect_source(refined_query)
        assert source == "jira"

    def test_detect_jira_from_keywords(self, router):
        """keywords에서 jira 키워드 감지."""
        refined_query = RefinedQuery(
            original_query="스프린트 현황",
            refined_query="스프린트 검색",
            keywords=["issue", "sprint"],
        )
        source = router._detect_source(refined_query)
        assert source == "jira"

    def test_detect_no_source(self, router):
        """소스 힌트가 없는 경우 None 반환."""
        refined_query = RefinedQuery(
            original_query="배포 현황 알려줘",
            refined_query="배포 현황 검색",
            keywords=["배포", "현황"],
        )
        source = router._detect_source(refined_query)
        assert source is None

    def test_detect_source_case_insensitive(self, router):
        """소스 감지는 대소문자 구분하지 않음."""
        refined_query = RefinedQuery(
            original_query="SLACK 채널에서",
            refined_query="채널 검색",
            keywords=["CHANNEL"],
        )
        source = router._detect_source(refined_query)
        assert source == "slack"


class TestSearchPlan:
    """SearchPlan 데이터클래스 테스트."""

    def test_default_values(self):
        """기본값 테스트."""
        plan = SearchPlan(queries=["test query"])
        assert plan.strategy == SearchStrategy.HYBRID
        assert plan.source_filter is None
        assert plan.metadata_filters == {}

    def test_all_fields(self):
        """모든 필드 설정 테스트."""
        plan = SearchPlan(
            queries=["query1", "query2"],
            strategy=SearchStrategy.SPARSE_ONLY,
            source_filter="jira",
            metadata_filters={"author": "Alice"},
        )
        assert len(plan.queries) == 2
        assert plan.strategy == SearchStrategy.SPARSE_ONLY
        assert plan.source_filter == "jira"
        assert plan.metadata_filters["author"] == "Alice"


class TestSearchStrategy:
    """SearchStrategy enum 테스트."""

    def test_strategy_values(self):
        """전략 값 확인."""
        assert SearchStrategy.HYBRID.value == "hybrid"
        assert SearchStrategy.DENSE_ONLY.value == "dense_only"
        assert SearchStrategy.SPARSE_ONLY.value == "sparse_only"
