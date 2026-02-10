"""SlackAdapter 테스트 - Slack API 모킹."""

import pytest

from src.ingestion.adapters.base import DocumentACL
from src.ingestion.adapters.slack import SlackAdapter


class TestSlackAdapter:
    """SlackAdapter 테스트."""

    @pytest.fixture
    def adapter(self):
        """SlackAdapter 인스턴스."""
        return SlackAdapter()

    def test_source_name(self, adapter):
        """source_name이 'slack'인지 확인."""
        assert adapter.source_name == "slack"

    def test_fetch_updates_returns_documents(self, adapter, mock_slack_api):
        """fetch_updates가 RawDocument 리스트를 반환."""
        documents = adapter.fetch_updates()

        assert len(documents) > 0
        for doc in documents:
            assert doc.source == "slack"
            assert doc.doc_id.startswith("slack_")

    def test_fetch_updates_includes_channel_info(self, adapter, mock_slack_api):
        """문서에 채널 정보가 포함되어야 함."""
        documents = adapter.fetch_updates()

        for doc in documents:
            assert "channel_id" in doc.metadata
            assert "channel_name" in doc.metadata

    def test_fetch_updates_includes_thread_replies(self, adapter, mock_slack_api):
        """스레드 답글이 본문에 포함되어야 함."""
        documents = adapter.fetch_updates()

        # 답글이 있는 메시지는 "[Thread Replies]" 섹션이 포함될 수 있음
        has_thread = any("[Thread Replies]" in doc.content for doc in documents)
        # 테스트 데이터에 답글이 있으므로 True여야 함
        assert has_thread

    def test_extract_permissions_public_channel(self, adapter):
        """public 채널의 ACL 테스트."""
        raw_data = {
            "channel": {"id": "C123", "is_private": False},
            "members": ["U001", "U002"],
        }
        acl = adapter.extract_permissions(raw_data)

        assert acl.level == "public"
        assert acl.users == ["U001", "U002"]

    def test_extract_permissions_private_channel(self, adapter):
        """private 채널의 ACL 테스트."""
        raw_data = {
            "channel": {"id": "C456", "is_private": True},
            "members": ["U001"],
        }
        acl = adapter.extract_permissions(raw_data)

        assert acl.level == "private"
        assert acl.users == ["U001"]

    def test_document_url_format(self, adapter, mock_slack_api):
        """문서 URL 포맷 확인."""
        documents = adapter.fetch_updates()

        for doc in documents:
            assert doc.url.startswith("https://slack.com/archives/")

    def test_list_channels_api_error(self, adapter):
        """API 에러 시 빈 리스트 반환."""
        import respx
        from httpx import Response

        with respx.mock(assert_all_called=False) as respx_mock:
            respx_mock.get("https://slack.com/api/conversations.list").mock(
                return_value=Response(200, json={"ok": False, "error": "invalid_auth"})
            )

            channels = adapter._list_channels()
            assert channels == []

    def test_fetch_updates_empty_channels(self, adapter):
        """채널이 없을 때 빈 리스트 반환."""
        import respx
        from httpx import Response

        with respx.mock(assert_all_called=False) as respx_mock:
            respx_mock.get("https://slack.com/api/conversations.list").mock(
                return_value=Response(200, json={"ok": True, "channels": []})
            )

            documents = adapter.fetch_updates()
            assert documents == []
