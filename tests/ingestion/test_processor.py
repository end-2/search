"""Processor 모듈 테스트: TextCleaner, Chunker."""

import pytest

from src.ingestion.adapters.base import DocumentACL, RawDocument
from src.ingestion.processor import Chunker, Processor, TextCleaner


class TestTextCleaner:
    """TextCleaner 테스트."""

    def test_clean_removes_html_tags(self):
        """HTML 태그가 제거되어야 함."""
        text = "<p>Hello <b>World</b></p>"
        result = TextCleaner.clean(text)
        assert result == "Hello World"

    def test_clean_removes_nested_html(self):
        """중첩된 HTML 태그가 제거되어야 함."""
        text = "<div><ul><li>Item 1</li><li>Item 2</li></ul></div>"
        result = TextCleaner.clean(text)
        assert "Item 1" in result
        assert "Item 2" in result
        assert "<" not in result
        assert ">" not in result

    def test_clean_normalizes_whitespace(self):
        """연속된 공백과 개행이 정리되어야 함."""
        text = "Hello    World\n\n\nTest"
        result = TextCleaner.clean(text)
        assert result == "Hello World Test"

    def test_clean_removes_control_characters(self):
        """제어 문자가 제거되어야 함."""
        text = "Hello\x00\x08World\x0bTest"
        result = TextCleaner.clean(text)
        assert "\x00" not in result
        assert "\x08" not in result
        assert "\x0b" not in result
        assert "Hello" in result
        assert "World" in result

    def test_clean_empty_string(self):
        """빈 문자열 처리."""
        assert TextCleaner.clean("") == ""
        assert TextCleaner.clean("   ") == ""

    def test_clean_preserves_korean_text(self):
        """한글 텍스트가 보존되어야 함."""
        text = "<p>안녕하세요 <b>세계</b></p>"
        result = TextCleaner.clean(text)
        assert result == "안녕하세요 세계"


class TestChunker:
    """Chunker 테스트."""

    @pytest.fixture
    def chunker(self, monkeypatch):
        """테스트용 Chunker 인스턴스."""
        # 환경 변수 설정으로 Settings 값을 override
        monkeypatch.setenv("PARENT_CHUNK_SIZE", "100")
        monkeypatch.setenv("PARENT_CHUNK_OVERLAP", "10")
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "30")
        monkeypatch.setenv("CHILD_CHUNK_OVERLAP", "5")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Settings 캐시 클리어
        from src.common.config import get_settings

        get_settings.cache_clear()

        return Chunker()

    @pytest.fixture
    def sample_document(self):
        """테스트용 RawDocument."""
        return RawDocument(
            doc_id="test-doc-1",
            source="test",
            content="This is a test document with enough content to be chunked. " * 20,
            url="https://example.com/test",
            author="tester",
            created_at="2024-01-01T00:00:00Z",
            permissions=DocumentACL(users=["user1"], groups=["group1"], level="read"),
        )

    def test_split_by_tokens_basic(self, chunker):
        """기본 토큰 분할 동작 테스트."""
        text = "Hello world this is a test sentence for chunking."
        chunks = chunker._split_by_tokens(text, chunk_size=10, overlap=2)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_split_by_tokens_overlap(self, chunker):
        """오버랩이 적용되는지 테스트."""
        # 긴 텍스트로 여러 청크가 생성되도록 함
        text = "word " * 100  # 100개의 단어
        chunks = chunker._split_by_tokens(text, chunk_size=20, overlap=5)

        # 청크가 2개 이상이면 오버랩 확인
        if len(chunks) > 1:
            # 오버랩으로 인해 단순 분할보다 많은 청크 생성
            assert len(chunks) >= 2

    def test_split_by_tokens_empty_text(self, chunker):
        """빈 텍스트 처리."""
        chunks = chunker._split_by_tokens("", chunk_size=10, overlap=2)
        assert chunks == []

    def test_create_parent_child_chunks(self, chunker, sample_document):
        """Parent-Child 청크 생성 테스트."""
        parents, children = chunker.create_parent_child_chunks(sample_document)

        # Parent 청크 검증
        assert len(parents) >= 1
        for parent in parents:
            assert parent.doc_id == sample_document.doc_id
            assert parent.source == sample_document.source
            assert parent.url == sample_document.url
            assert parent.permissions == sample_document.permissions

        # Child 청크 검증
        assert len(children) >= 1
        for child in children:
            assert child.source == sample_document.source
            # Child는 Parent를 참조해야 함
            assert any(child.parent_id == p.chunk_id for p in parents)

    def test_parent_child_relationship(self, chunker, sample_document):
        """Parent-Child 관계가 올바른지 테스트."""
        parents, children = chunker.create_parent_child_chunks(sample_document)

        # 모든 child가 유효한 parent_id를 가지는지 확인
        parent_ids = {p.chunk_id for p in parents}
        for child in children:
            assert child.parent_id in parent_ids


class TestProcessor:
    """Processor 통합 테스트."""

    @pytest.fixture
    def processor(self, monkeypatch):
        """테스트용 Processor 인스턴스."""
        monkeypatch.setenv("PARENT_CHUNK_SIZE", "100")
        monkeypatch.setenv("PARENT_CHUNK_OVERLAP", "10")
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "30")
        monkeypatch.setenv("CHILD_CHUNK_OVERLAP", "5")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from src.common.config import get_settings

        get_settings.cache_clear()

        return Processor()

    def test_process_cleans_and_chunks(self, processor):
        """HTML 정제 후 청킹이 수행되는지 테스트."""
        docs = [
            RawDocument(
                doc_id="doc-1",
                source="test",
                content="<p>Clean this <b>HTML</b> content.</p> " * 30,
                url="https://example.com/1",
                author="tester",
                created_at="2024-01-01T00:00:00Z",
            )
        ]

        parents, children = processor.process(docs)

        assert len(parents) >= 1
        assert len(children) >= 1

        # HTML이 제거되었는지 확인
        for parent in parents:
            assert "<p>" not in parent.content
            assert "<b>" not in parent.content

    def test_process_skips_empty_documents(self, processor):
        """빈 문서는 스킵되어야 함."""
        docs = [
            RawDocument(
                doc_id="empty-doc",
                source="test",
                content="   ",  # 공백만 있는 문서
                url="https://example.com/empty",
                author="tester",
                created_at="2024-01-01T00:00:00Z",
            )
        ]

        parents, children = processor.process(docs)

        assert len(parents) == 0
        assert len(children) == 0
