"""VectorStore 모듈 테스트 - Qdrant 모킹."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.adapters.base import DocumentACL
from src.ingestion.processor import ChildChunk, ParentChunk
from src.ingestion.vector_store import VectorStore


class TestVectorStore:
    """VectorStore 테스트."""

    @pytest.fixture
    def vector_store(self, mock_qdrant_client):
        """VectorStore 인스턴스 (Qdrant 모킹됨)."""
        return VectorStore()

    def test_init_creates_client(self, mock_qdrant_client):
        """VectorStore 초기화 시 Qdrant 클라이언트 생성."""
        store = VectorStore()
        assert store._client is not None

    def test_init_collections_creates_both_collections(self, vector_store, mock_qdrant_client):
        """init_collections가 child와 parent 컬렉션을 모두 생성."""
        vector_store.init_collections()

        # create_collection이 2번 호출됨 (child, parent)
        assert mock_qdrant_client.create_collection.call_count == 2

    def test_init_collections_skips_existing(self, mock_qdrant_client):
        """이미 존재하는 컬렉션은 생성하지 않음."""
        # 컬렉션이 이미 존재하는 것으로 설정
        mock_collection = MagicMock()
        mock_collection.name = "child_chunks"
        mock_collection2 = MagicMock()
        mock_collection2.name = "parent_chunks"
        mock_qdrant_client.get_collections.return_value.collections = [
            mock_collection,
            mock_collection2,
        ]

        store = VectorStore()
        store.init_collections()

        # create_collection이 호출되지 않음
        mock_qdrant_client.create_collection.assert_not_called()

    def test_upsert_parents(self, vector_store, mock_qdrant_client, sample_parent_chunk):
        """Parent Chunk upsert 테스트."""
        parents = [sample_parent_chunk]
        vector_store.upsert_parents(parents)

        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args[1]["collection_name"] == "parent_chunks"
        assert len(call_args[1]["points"]) == 1

    def test_upsert_children(self, vector_store, mock_qdrant_client, sample_child_chunks):
        """Child Chunk upsert 테스트."""
        children = sample_child_chunks[:2]
        dense_vectors = [[0.1] * 1536, [0.2] * 1536]
        sparse_vectors = [([0, 1], [0.5, 0.3]), ([1, 2], [0.4, 0.6])]

        vector_store.upsert_children(children, dense_vectors, sparse_vectors)

        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args[1]["collection_name"] == "child_chunks"
        assert len(call_args[1]["points"]) == 2

    def test_upsert_children_with_permissions(self, vector_store, mock_qdrant_client):
        """Child Chunk upsert 시 ACL 정보 포함 확인."""
        child = ChildChunk(
            chunk_id="test-child",
            parent_id="test-parent",
            content="Test content",
            source="test",
            url="https://example.com",
            author="tester",
            permissions=DocumentACL(users=["user1"], groups=["group1"], level="private"),
        )
        dense_vectors = [[0.1] * 1536]
        sparse_vectors = [([0], [0.5])]

        vector_store.upsert_children([child], dense_vectors, sparse_vectors)

        call_args = mock_qdrant_client.upsert.call_args
        point = call_args[1]["points"][0]
        assert point.payload["permissions"]["users"] == ["user1"]
        assert point.payload["permissions"]["groups"] == ["group1"]
        assert point.payload["permissions"]["level"] == "private"

    def test_get_parent_by_id_found(self, vector_store, mock_qdrant_client):
        """Parent ID로 조회 성공."""
        mock_result = MagicMock()
        mock_result.payload = {
            "doc_id": "doc-1",
            "content": "Parent content",
            "source": "test",
        }
        mock_qdrant_client.retrieve.return_value = [mock_result]

        result = vector_store.get_parent_by_id("parent-1")

        assert result is not None
        assert result["content"] == "Parent content"
        mock_qdrant_client.retrieve.assert_called_once()

    def test_get_parent_by_id_not_found(self, vector_store, mock_qdrant_client):
        """Parent ID로 조회 실패."""
        mock_qdrant_client.retrieve.return_value = []

        result = vector_store.get_parent_by_id("non-existent")

        assert result is None

    def test_search_children_basic(self, vector_store, mock_qdrant_client):
        """기본 검색 테스트."""
        mock_hit = MagicMock()
        mock_hit.id = "child-1"
        mock_hit.score = 0.95
        mock_hit.payload = {"content_text": "Test content"}
        mock_qdrant_client.search.return_value = [mock_hit]

        results = vector_store.search_children(
            dense_vector=[0.1] * 1536,
            user_id="user1",
            top_k=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "child-1"
        assert results[0]["score"] == 0.95

    def test_search_children_with_acl_filter(self, vector_store, mock_qdrant_client):
        """ACL 필터가 포함된 검색."""
        vector_store.search_children(
            dense_vector=[0.1] * 1536,
            user_id="alice",
            user_groups=["engineering", "team"],
            top_k=50,
        )

        mock_qdrant_client.search.assert_called_once()
        call_args = mock_qdrant_client.search.call_args
        # query_filter가 전달되었는지 확인
        assert call_args[1]["query_filter"] is not None

    def test_batch_upsert_large_batch(self, vector_store, mock_qdrant_client):
        """150개 포인트가 2배치로 분할되는지 테스트."""
        parents = [
            ParentChunk(
                chunk_id=f"parent-{i}",
                doc_id=f"doc-{i}",
                content=f"Content {i}",
                source="test",
                url="https://example.com",
                author="tester",
            )
            for i in range(150)
        ]

        vector_store.upsert_parents(parents)

        # 100 + 50으로 2번 호출
        assert mock_qdrant_client.upsert.call_count == 2
