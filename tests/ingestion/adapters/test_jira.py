"""JiraAdapter 테스트 - Jira API 모킹."""

import pytest

from src.ingestion.adapters.base import DocumentACL
from src.ingestion.adapters.jira import JiraAdapter


class TestJiraAdapter:
    """JiraAdapter 테스트."""

    @pytest.fixture
    def adapter(self):
        """JiraAdapter 인스턴스."""
        return JiraAdapter()

    def test_source_name(self, adapter):
        """source_name이 'jira'인지 확인."""
        assert adapter.source_name == "jira"

    def test_fetch_updates_returns_documents(self, adapter, mock_jira_api):
        """fetch_updates가 RawDocument 리스트를 반환."""
        documents = adapter.fetch_updates()

        assert len(documents) == 1
        doc = documents[0]
        assert doc.source == "jira"
        assert doc.doc_id.startswith("jira_")

    def test_fetch_updates_document_structure(self, adapter, mock_jira_api):
        """문서 구조 확인."""
        documents = adapter.fetch_updates()
        doc = documents[0]

        assert "TEST-1" in doc.doc_id
        assert "Test Issue" in doc.title
        assert doc.author == "Bob"
        assert "project_key" in doc.metadata
        assert doc.metadata["project_key"] == "TEST"

    def test_fetch_updates_includes_description(self, adapter, mock_jira_api):
        """이슈 설명이 본문에 포함되어야 함."""
        documents = adapter.fetch_updates()
        doc = documents[0]

        assert "Issue description" in doc.content

    def test_extract_permissions(self, adapter):
        """프로젝트 기반 ACL 테스트."""
        raw_data = {
            "project_key": "TEST",
            "assignee": "alice@example.com",
            "reporter": "bob@example.com",
        }
        acl = adapter.extract_permissions(raw_data)

        assert acl.level == "private"
        assert "alice@example.com" in acl.users
        assert "bob@example.com" in acl.users
        assert "jira-test" in acl.groups

    def test_extract_permissions_no_assignee(self, adapter):
        """담당자가 없는 경우."""
        raw_data = {
            "project_key": "PROJ",
            "assignee": "",
            "reporter": "reporter@example.com",
        }
        acl = adapter.extract_permissions(raw_data)

        assert "reporter@example.com" in acl.users
        assert "" not in acl.users  # 빈 문자열은 제외

    def test_document_url_format(self, adapter, mock_jira_api):
        """문서 URL 포맷 확인."""
        documents = adapter.fetch_updates()
        doc = documents[0]

        assert doc.url == "https://test.atlassian.net/browse/TEST-1"

    def test_extract_text_from_adf(self, adapter):
        """ADF 형식에서 텍스트 추출."""
        adf_content = {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Hello "}]},
                {"type": "paragraph", "content": [{"type": "text", "text": "World"}]},
            ],
        }
        text = adapter._extract_text(adf_content)

        assert "Hello" in text
        assert "World" in text

    def test_extract_text_from_string(self, adapter):
        """문자열 형식 설명 처리."""
        text = adapter._extract_text("Simple string description")
        assert text == "Simple string description"

    def test_extract_text_none(self, adapter):
        """None 설명 처리."""
        text = adapter._extract_text(None)
        assert text == ""

    def test_extract_comments(self, adapter):
        """코멘트 추출 테스트."""
        comment_data = {
            "comments": [
                {
                    "author": {"displayName": "Alice"},
                    "body": {"type": "doc", "content": [{"type": "text", "text": "Comment 1"}]},
                },
                {
                    "author": {"displayName": "Bob"},
                    "body": "Comment 2",
                },
            ]
        }
        comments = adapter._extract_comments(comment_data)

        assert "Alice" in comments
        assert "Comment 1" in comments
        assert "Bob" in comments
        assert "Comment 2" in comments

    def test_search_issues_api_error(self, adapter):
        """API 에러 시 빈 리스트 반환."""
        import respx
        from httpx import Response

        with respx.mock(assert_all_called=False) as respx_mock:
            respx_mock.get("https://test.atlassian.net/rest/api/3/search").mock(
                return_value=Response(401, json={"error": "Unauthorized"})
            )

            issues = adapter._search_issues("ORDER BY updated DESC", 0, 50)
            assert issues == []

    def test_fetch_updates_with_last_sync_time(self, adapter):
        """last_sync_time이 JQL에 포함되어야 함."""
        import respx
        from datetime import datetime
        from httpx import Response

        with respx.mock(assert_all_called=False) as respx_mock:
            route = respx_mock.get("https://test.atlassian.net/rest/api/3/search").mock(
                return_value=Response(200, json={"issues": []})
            )

            last_sync = datetime(2024, 1, 15, 10, 30)
            adapter.fetch_updates(last_sync_time=last_sync)

            # JQL에 날짜가 포함되었는지 확인
            request = route.calls[0].request
            assert "2024-01-15" in str(request.url)
