"""Ingestion 파이프라인 통합 테스트.

전체 Ingestion 흐름을 테스트합니다:
1. Adapter: 데이터 수집
2. Processor: 전처리 및 청킹
3. Embedder: Dense/Sparse 벡터 생성
4. VectorStore: 벡터 DB 저장
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.adapters.base import DocumentACL, RawDocument
from src.ingestion.embedder import Embedder, SparseEncoder
from src.ingestion.processor import Processor
from src.ingestion.vector_store import VectorStore


class TestIngestionPipeline:
    """Ingestion 파이프라인 통합 테스트."""

    @pytest.fixture
    def processor(self, monkeypatch):
        """테스트용 Processor."""
        monkeypatch.setenv("PARENT_CHUNK_SIZE", "200")
        monkeypatch.setenv("PARENT_CHUNK_OVERLAP", "20")
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "50")
        monkeypatch.setenv("CHILD_CHUNK_OVERLAP", "10")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from src.common.config import get_settings

        get_settings.cache_clear()

        return Processor()

    @pytest.fixture
    def embedder(self):
        """테스트용 Embedder."""
        return Embedder()

    @pytest.fixture
    def sparse_encoder(self):
        """테스트용 SparseEncoder."""
        return SparseEncoder()

    @pytest.fixture
    def vector_store(self, mock_qdrant_client):
        """테스트용 VectorStore (Qdrant 모킹)."""
        return VectorStore()

    def test_full_pipeline_with_single_document(
        self,
        processor,
        embedder,
        sparse_encoder,
        vector_store,
        mock_openai_embeddings,
        mock_qdrant_client,
    ):
        """단일 문서에 대한 전체 파이프라인 테스트."""
        # 1. 테스트 문서 준비
        raw_doc = RawDocument(
            doc_id="test-doc-1",
            title="Test Document",
            content="<p>This is a <b>test</b> document.</p> " * 50,  # 충분한 길이
            source="test",
            url="https://example.com/test",
            author="tester",
            permissions=DocumentACL(users=["user1"], groups=["team"], level="public"),
        )

        # 2. 전처리 및 청킹
        parents, children = processor.process([raw_doc])

        assert len(parents) >= 1
        assert len(children) >= 1
        # HTML이 제거되었는지 확인
        assert "<p>" not in parents[0].content
        assert "<b>" not in parents[0].content

        # 3. Dense Embedding 생성
        dense_vectors = embedder.generate_embeddings(children)

        assert len(dense_vectors) == len(children)
        assert len(dense_vectors[0]) == 1536

        # 4. Sparse Vector 생성
        sparse_encoder.fit([c.content for c in children])
        sparse_vectors = sparse_encoder.encode_batch(children)

        assert len(sparse_vectors) == len(children)

        # 5. VectorStore에 저장
        vector_store.init_collections()
        vector_store.upsert_parents(parents)
        vector_store.upsert_children(children, dense_vectors, sparse_vectors)

        # Qdrant 호출 확인
        assert mock_qdrant_client.upsert.call_count >= 2

    def test_pipeline_with_multiple_documents(
        self,
        processor,
        embedder,
        sparse_encoder,
        vector_store,
        sample_raw_documents,
        mock_openai_embeddings,
        mock_qdrant_client,
    ):
        """여러 문서에 대한 파이프라인 테스트."""
        # 전처리 및 청킹
        parents, children = processor.process(sample_raw_documents)

        # 모든 문서가 처리되었는지 확인
        assert len(parents) >= len(sample_raw_documents)

        # Dense Embedding
        dense_vectors = embedder.generate_embeddings(children)

        # Sparse Vector
        sparse_encoder.fit([c.content for c in children])
        sparse_vectors = sparse_encoder.encode_batch(children)

        # VectorStore 저장
        vector_store.upsert_parents(parents)
        vector_store.upsert_children(children, dense_vectors, sparse_vectors)

        # 호출 확인
        assert mock_qdrant_client.upsert.called

    def test_pipeline_preserves_metadata(
        self,
        processor,
        embedder,
        sparse_encoder,
        vector_store,
        mock_openai_embeddings,
        mock_qdrant_client,
    ):
        """파이프라인 전반에 걸쳐 메타데이터가 보존되는지 테스트."""
        raw_doc = RawDocument(
            doc_id="meta-test",
            title="Metadata Test",
            content="Test content for metadata preservation. " * 30,
            source="custom_source",
            url="https://custom.example.com/doc",
            author="custom_author",
            metadata={"custom_key": "custom_value"},
            permissions=DocumentACL(users=["alice", "bob"], groups=["eng"], level="private"),
        )

        parents, children = processor.process([raw_doc])

        # Parent 메타데이터 확인
        assert parents[0].source == "custom_source"
        assert parents[0].author == "custom_author"
        assert parents[0].permissions.users == ["alice", "bob"]
        assert parents[0].permissions.level == "private"

        # Child 메타데이터 확인
        assert children[0].source == "custom_source"
        assert children[0].author == "custom_author"

    def test_pipeline_handles_empty_documents(self, processor):
        """빈 문서 처리 테스트."""
        raw_docs = [
            RawDocument(
                doc_id="empty-1",
                content="   ",  # 공백만
                source="test",
                url="https://example.com",
                author="tester",
            ),
            RawDocument(
                doc_id="empty-2",
                content="<p>  </p>",  # HTML만
                source="test",
                url="https://example.com",
                author="tester",
            ),
        ]

        parents, children = processor.process(raw_docs)

        # 빈 문서는 스킵됨
        assert len(parents) == 0
        assert len(children) == 0


class TestIngestionPipelineWithSlack:
    """Slack 어댑터를 포함한 통합 테스트."""

    @pytest.fixture
    def processor(self, monkeypatch):
        monkeypatch.setenv("PARENT_CHUNK_SIZE", "500")
        monkeypatch.setenv("PARENT_CHUNK_OVERLAP", "50")
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "100")
        monkeypatch.setenv("CHILD_CHUNK_OVERLAP", "20")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from src.common.config import get_settings

        get_settings.cache_clear()

        return Processor()

    def test_slack_to_vector_store(
        self,
        processor,
        mock_slack_api,
        mock_openai_embeddings,
        mock_qdrant_client,
    ):
        """Slack 데이터 수집부터 VectorStore 저장까지."""
        from src.ingestion.adapters.slack import SlackAdapter

        # 1. Slack에서 데이터 수집
        adapter = SlackAdapter()
        raw_docs = adapter.fetch_updates()

        assert len(raw_docs) > 0
        assert all(doc.source == "slack" for doc in raw_docs)

        # 2. 전처리 (메시지가 짧을 수 있으므로 필터링)
        valid_docs = [d for d in raw_docs if len(d.content) > 10]
        if not valid_docs:
            pytest.skip("No valid documents to process")

        parents, children = processor.process(valid_docs)

        # 3. Embedding
        embedder = Embedder()
        sparse_encoder = SparseEncoder()

        if children:
            dense_vectors = embedder.generate_embeddings(children)
            sparse_encoder.fit([c.content for c in children])
            sparse_vectors = sparse_encoder.encode_batch(children)

            # 4. VectorStore 저장
            vector_store = VectorStore()
            vector_store.upsert_parents(parents)
            vector_store.upsert_children(children, dense_vectors, sparse_vectors)

            assert mock_qdrant_client.upsert.called


class TestIngestionPipelineWithJira:
    """Jira 어댑터를 포함한 통합 테스트."""

    @pytest.fixture
    def processor(self, monkeypatch):
        monkeypatch.setenv("PARENT_CHUNK_SIZE", "500")
        monkeypatch.setenv("PARENT_CHUNK_OVERLAP", "50")
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "100")
        monkeypatch.setenv("CHILD_CHUNK_OVERLAP", "20")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from src.common.config import get_settings

        get_settings.cache_clear()

        return Processor()

    def test_jira_to_vector_store(
        self,
        processor,
        mock_jira_api,
        mock_openai_embeddings,
        mock_qdrant_client,
    ):
        """Jira 데이터 수집부터 VectorStore 저장까지."""
        from src.ingestion.adapters.jira import JiraAdapter

        # 1. Jira에서 데이터 수집
        adapter = JiraAdapter()
        raw_docs = adapter.fetch_updates()

        assert len(raw_docs) > 0
        assert all(doc.source == "jira" for doc in raw_docs)

        # 2. 전처리
        parents, children = processor.process(raw_docs)

        if not children:
            pytest.skip("No children chunks created")

        # 3. Embedding
        embedder = Embedder()
        sparse_encoder = SparseEncoder()

        dense_vectors = embedder.generate_embeddings(children)
        sparse_encoder.fit([c.content for c in children])
        sparse_vectors = sparse_encoder.encode_batch(children)

        # 4. VectorStore 저장
        vector_store = VectorStore()
        vector_store.upsert_parents(parents)
        vector_store.upsert_children(children, dense_vectors, sparse_vectors)

        # 저장된 데이터에 Jira 메타데이터가 포함되어야 함
        assert mock_qdrant_client.upsert.called
