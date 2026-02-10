"""공통 테스트 픽스처."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.adapters.base import DocumentACL, RawDocument
from src.ingestion.processor import ChildChunk, ParentChunk


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """테스트 환경 설정."""
    # 테스트에 필요한 기본 환경 변수 설정
    os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
    os.environ.setdefault("QDRANT_HOST", "localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")
    os.environ.setdefault("MYSQL_HOST", "localhost")
    os.environ.setdefault("MYSQL_PORT", "3306")
    os.environ.setdefault("MYSQL_USER", "test")
    os.environ.setdefault("MYSQL_PASSWORD", "test")
    os.environ.setdefault("MYSQL_DATABASE", "test_db")
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("DEBUG", "false")
    os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test-token")
    os.environ.setdefault("JIRA_BASE_URL", "https://test.atlassian.net")
    os.environ.setdefault("JIRA_EMAIL", "test@example.com")
    os.environ.setdefault("JIRA_API_TOKEN", "test-token")

    yield


@pytest.fixture
def clear_settings_cache():
    """Settings 캐시 클리어 픽스처."""
    from src.common.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- Sample Data Fixtures ---


@pytest.fixture
def sample_raw_document():
    """테스트용 RawDocument."""
    return RawDocument(
        doc_id="test-doc-1",
        title="Test Document",
        content="This is a test document with sufficient content for testing purposes. " * 10,
        source="test",
        url="https://example.com/test",
        author="tester",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        metadata={"key": "value"},
        permissions=DocumentACL(users=["user1"], groups=["group1"], level="public"),
    )


@pytest.fixture
def sample_raw_documents():
    """테스트용 RawDocument 리스트."""
    return [
        RawDocument(
            doc_id=f"test-doc-{i}",
            title=f"Test Document {i}",
            content=f"Content for document {i}. " * 20,
            source="test",
            url=f"https://example.com/test/{i}",
            author=f"author{i}",
            created_at=datetime(2024, 1, i + 1, 12, 0, 0),
            permissions=DocumentACL(users=[f"user{i}"], groups=["team"], level="public"),
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_parent_chunk():
    """테스트용 ParentChunk."""
    return ParentChunk(
        chunk_id="parent-1",
        doc_id="test-doc-1",
        content="This is a parent chunk with extended context for LLM generation.",
        source="test",
        url="https://example.com/test",
        author="tester",
        metadata={"key": "value"},
        permissions=DocumentACL(users=["user1"], groups=["group1"], level="public"),
    )


@pytest.fixture
def sample_child_chunk():
    """테스트용 ChildChunk."""
    return ChildChunk(
        chunk_id="child-1",
        parent_id="parent-1",
        content="This is a child chunk for vector search.",
        source="test",
        url="https://example.com/test",
        author="tester",
        metadata={"key": "value"},
        permissions=DocumentACL(users=["user1"], groups=["group1"], level="public"),
    )


@pytest.fixture
def sample_child_chunks():
    """테스트용 ChildChunk 리스트."""
    return [
        ChildChunk(
            chunk_id=f"child-{i}",
            parent_id="parent-1",
            content=f"Child chunk content number {i} for testing.",
            source="test",
            url="https://example.com/test",
            author="tester",
        )
        for i in range(5)
    ]


# --- OpenAI Mock Fixtures ---


@pytest.fixture
def mock_openai_embeddings():
    """OpenAI Embeddings API 모킹."""
    with patch("src.common.openai_utils.create_embeddings") as mock:
        # 1536차원 더미 벡터 반환
        mock.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
        yield mock


@pytest.fixture
def mock_openai_chat():
    """OpenAI Chat Completion API 모킹."""
    with patch("src.common.openai_utils.chat_completion") as mock:
        mock.return_value = "This is a mock response from the LLM."
        yield mock


@pytest.fixture
def mock_openai_client():
    """OpenAI Client 전체 모킹."""
    with patch("src.common.openai_utils.get_openai_client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


# --- Qdrant Mock Fixtures ---


@pytest.fixture
def mock_qdrant_client():
    """Qdrant Client 모킹."""
    with patch("src.ingestion.vector_store.QdrantClient") as mock_cls:
        client = MagicMock()

        # get_collections 모킹
        collections_response = MagicMock()
        collections_response.collections = []
        client.get_collections.return_value = collections_response

        # create_collection 모킹
        client.create_collection.return_value = True

        # upsert 모킹
        client.upsert.return_value = True

        # retrieve 모킹
        client.retrieve.return_value = []

        # search 모킹
        client.search.return_value = []

        mock_cls.return_value = client
        yield client


@pytest.fixture
def mock_qdrant_with_data(mock_qdrant_client):
    """검색 결과가 있는 Qdrant Client 모킹."""
    # search 결과 모킹
    mock_hit = MagicMock()
    mock_hit.id = "child-1"
    mock_hit.score = 0.95
    mock_hit.payload = {
        "parent_id": "parent-1",
        "content_text": "Test child content",
        "source": "test",
        "url": "https://example.com/test",
        "author": "tester",
        "permissions": {"users": ["user1"], "groups": ["group1"], "level": "public"},
    }
    mock_qdrant_client.search.return_value = [mock_hit]

    # retrieve 결과 모킹 (Parent)
    mock_parent = MagicMock()
    mock_parent.payload = {
        "doc_id": "test-doc-1",
        "content": "This is the parent chunk content with more context.",
        "source": "test",
        "url": "https://example.com/test",
        "author": "tester",
    }
    mock_qdrant_client.retrieve.return_value = [mock_parent]

    return mock_qdrant_client


# --- HTTP Mock Fixtures (for Adapters) ---


@pytest.fixture
def mock_slack_api():
    """Slack API 모킹."""
    import respx
    from httpx import Response

    with respx.mock(assert_all_called=False) as respx_mock:
        # conversations.list
        respx_mock.get("https://slack.com/api/conversations.list").mock(
            return_value=Response(
                200,
                json={
                    "ok": True,
                    "channels": [
                        {"id": "C123", "name": "general", "is_private": False},
                        {"id": "C456", "name": "dev", "is_private": True},
                    ],
                },
            )
        )

        # conversations.members
        respx_mock.get("https://slack.com/api/conversations.members").mock(
            return_value=Response(
                200,
                json={"ok": True, "members": ["U001", "U002", "U003"]},
            )
        )

        # conversations.history
        respx_mock.get("https://slack.com/api/conversations.history").mock(
            return_value=Response(
                200,
                json={
                    "ok": True,
                    "messages": [
                        {"ts": "1704067200.000000", "text": "Hello world!", "user": "U001"},
                        {"ts": "1704067300.000000", "text": "Test message", "user": "U002"},
                    ],
                },
            )
        )

        # conversations.replies
        respx_mock.get("https://slack.com/api/conversations.replies").mock(
            return_value=Response(
                200,
                json={
                    "ok": True,
                    "messages": [
                        {"ts": "1704067200.000000", "text": "Original"},
                        {"ts": "1704067201.000000", "text": "Reply 1"},
                    ],
                },
            )
        )

        yield respx_mock


@pytest.fixture
def mock_jira_api():
    """Jira API 모킹."""
    import respx
    from httpx import Response

    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.get("https://test.atlassian.net/rest/api/3/search").mock(
            return_value=Response(
                200,
                json={
                    "issues": [
                        {
                            "key": "TEST-1",
                            "fields": {
                                "summary": "Test Issue",
                                "description": {"type": "doc", "content": [{"type": "text", "text": "Issue description"}]},
                                "status": {"name": "Open"},
                                "project": {"key": "TEST"},
                                "assignee": {"displayName": "Alice", "emailAddress": "alice@example.com"},
                                "reporter": {"displayName": "Bob", "emailAddress": "bob@example.com"},
                                "created": "2024-01-01T12:00:00.000+0000",
                                "comment": {"comments": []},
                            },
                        }
                    ]
                },
            )
        )

        yield respx_mock


# --- UserContext Fixture ---


@pytest.fixture
def sample_user_context():
    """테스트용 UserContext."""
    from src.serving.core.security import UserContext

    return UserContext(user_id="test-user", groups=["engineering", "team"])
