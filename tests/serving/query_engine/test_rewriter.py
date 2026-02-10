"""QueryRewriter 모듈 테스트."""

import json
from unittest.mock import patch

import pytest

from src.serving.query_engine.rewriter import QueryRewriter, RefinedQuery


class TestRefinedQuery:
    """RefinedQuery 데이터클래스 테스트."""

    def test_default_values(self):
        """기본값 테스트."""
        query = RefinedQuery(
            original_query="test query",
            refined_query="refined test query",
        )
        assert query.sub_queries == []
        assert query.metadata_filters == {}
        assert query.keywords == []

    def test_all_fields(self):
        """모든 필드 설정 테스트."""
        query = RefinedQuery(
            original_query="original",
            refined_query="refined",
            sub_queries=["sub1", "sub2"],
            metadata_filters={"author": "Alice"},
            keywords=["keyword1", "keyword2"],
        )
        assert query.original_query == "original"
        assert query.refined_query == "refined"
        assert len(query.sub_queries) == 2
        assert query.metadata_filters["author"] == "Alice"
        assert len(query.keywords) == 2


class TestQueryRewriter:
    """QueryRewriter 테스트 - LLM API 모킹."""

    @pytest.fixture
    def rewriter(self):
        """QueryRewriter 인스턴스."""
        return QueryRewriter()

    def test_rewrite_successful(self, rewriter, mock_openai_chat):
        """정상적인 쿼리 리라이트 테스트."""
        mock_response = json.dumps({
            "refined_query": "배포 과정에서 발생한 에러 및 장애 내역",
            "sub_queries": [],
            "metadata_filters": {},
            "keywords": ["배포", "에러", "deploy", "error"],
        })
        mock_openai_chat.return_value = mock_response

        result = rewriter.rewrite("지난주 배포 에러 뭐였지?")

        assert isinstance(result, RefinedQuery)
        assert result.original_query == "지난주 배포 에러 뭐였지?"
        assert result.refined_query == "배포 과정에서 발생한 에러 및 장애 내역"
        assert "배포" in result.keywords
        mock_openai_chat.assert_called_once()

    def test_rewrite_with_sub_queries(self, rewriter, mock_openai_chat):
        """복합 질문 분해 테스트."""
        mock_response = json.dumps({
            "refined_query": "HR 정책 및 관련 티켓",
            "sub_queries": ["HR 정책 변경사항", "HR 관련 Jira 티켓"],
            "metadata_filters": {"author": "Alice"},
            "keywords": ["HR", "정책", "Jira"],
        })
        mock_openai_chat.return_value = mock_response

        result = rewriter.rewrite("Alice가 작성한 HR 정책과 Jira 티켓")

        assert len(result.sub_queries) == 2
        assert result.metadata_filters.get("author") == "Alice"

    def test_rewrite_json_parse_error_fallback(self, rewriter, mock_openai_chat):
        """JSON 파싱 실패 시 원본 쿼리 반환."""
        mock_openai_chat.return_value = "This is not valid JSON"

        result = rewriter.rewrite("테스트 쿼리")

        assert result.original_query == "테스트 쿼리"
        assert result.refined_query == "테스트 쿼리"
        # 공백으로 분리된 키워드
        assert "테스트" in result.keywords
        assert "쿼리" in result.keywords

    def test_rewrite_empty_response_fields(self, rewriter, mock_openai_chat):
        """응답에 일부 필드가 없는 경우."""
        mock_response = json.dumps({
            "refined_query": "refined query",
            # sub_queries, metadata_filters, keywords 누락
        })
        mock_openai_chat.return_value = mock_response

        result = rewriter.rewrite("original query")

        assert result.refined_query == "refined query"
        assert result.sub_queries == []
        assert result.metadata_filters == {}
        assert result.keywords == []

    def test_rewrite_preserves_original_query(self, rewriter, mock_openai_chat):
        """원본 쿼리가 항상 보존되어야 함."""
        mock_response = json.dumps({
            "refined_query": "다른 쿼리",
            "sub_queries": [],
            "metadata_filters": {},
            "keywords": [],
        })
        mock_openai_chat.return_value = mock_response

        original = "원본 질문입니다"
        result = rewriter.rewrite(original)

        assert result.original_query == original

    def test_rewrite_calls_chat_with_correct_messages(self, rewriter, mock_openai_chat):
        """chat_completion이 올바른 메시지로 호출되는지 확인."""
        mock_openai_chat.return_value = json.dumps({
            "refined_query": "test",
            "sub_queries": [],
            "metadata_filters": {},
            "keywords": [],
        })

        rewriter.rewrite("my test query")

        call_args = mock_openai_chat.call_args
        messages = call_args[0][0]  # 첫 번째 위치 인자

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "my test query"
        assert call_args[1]["temperature"] == 0.0
