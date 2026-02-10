"""Embedder 모듈 테스트: Embedder (Dense) 및 SparseEncoder (BM25)."""

import math
from unittest.mock import patch

import pytest

from src.ingestion.embedder import Embedder, SparseEncoder


class TestSparseEncoder:
    """SparseEncoder (BM25) 테스트."""

    @pytest.fixture
    def encoder(self):
        """기본 SparseEncoder 인스턴스."""
        return SparseEncoder(k1=1.5, b=0.75)

    @pytest.fixture
    def fitted_encoder(self, encoder):
        """학습된 SparseEncoder 인스턴스."""
        corpus = [
            "the quick brown fox jumps over the lazy dog",
            "the lazy dog sleeps all day",
            "the quick rabbit runs fast",
            "a fox is a clever animal",
        ]
        encoder.fit(corpus)
        return encoder

    def test_tokenize_basic(self):
        """기본 토큰화 테스트."""
        tokens = SparseEncoder._tokenize("Hello, World! How are you?")
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_tokenize_punctuation_removal(self):
        """구두점이 제거되어야 함."""
        tokens = SparseEncoder._tokenize("test... (hello) [world] {foo}")
        assert "test" in tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        # 구두점 단독은 제거됨
        assert "..." not in tokens

    def test_tokenize_lowercasing(self):
        """소문자로 변환되어야 함."""
        tokens = SparseEncoder._tokenize("Hello WORLD TeSt")
        assert all(t.islower() for t in tokens)

    def test_tokenize_empty_string(self):
        """빈 문자열 처리."""
        tokens = SparseEncoder._tokenize("")
        assert tokens == []

    def test_fit_builds_vocabulary(self, fitted_encoder):
        """fit 후 vocabulary가 구축되어야 함."""
        assert len(fitted_encoder._vocab) > 0
        assert "dog" in fitted_encoder._vocab
        assert "fox" in fitted_encoder._vocab

    def test_fit_calculates_idf(self, fitted_encoder):
        """fit 후 IDF 값이 계산되어야 함."""
        assert len(fitted_encoder._idf) > 0
        # 자주 나오는 단어는 낮은 IDF
        # "the"는 모든 문서에 나타남 → 낮은 IDF
        assert fitted_encoder._idf.get("the", 0) < fitted_encoder._idf.get("rabbit", float("inf"))

    def test_fit_calculates_avg_dl(self, fitted_encoder):
        """fit 후 평균 문서 길이가 계산되어야 함."""
        assert fitted_encoder._avg_dl > 0

    def test_encode_returns_sparse_vector(self, fitted_encoder):
        """encode가 (indices, values) 형태의 sparse vector를 반환해야 함."""
        indices, values = fitted_encoder.encode("the quick fox")

        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values)
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(v, float) for v in values)

    def test_encode_values_are_positive(self, fitted_encoder):
        """encode된 값들은 양수여야 함 (BM25 score)."""
        indices, values = fitted_encoder.encode("dog lazy fox")
        assert all(v > 0 for v in values)

    def test_encode_unknown_terms_ignored(self, fitted_encoder):
        """학습되지 않은 단어는 무시되어야 함."""
        indices, values = fitted_encoder.encode("xyz123 unknown terms")
        # unknown terms만 있으면 빈 결과
        assert len(indices) == 0
        assert len(values) == 0

    def test_encode_partial_match(self, fitted_encoder):
        """일부 단어만 매칭되는 경우."""
        # "dog"은 학습됨, "xyz"는 아님
        indices, values = fitted_encoder.encode("dog xyz")
        assert len(indices) == 1
        assert len(values) == 1

    def test_encode_empty_text(self, fitted_encoder):
        """빈 텍스트 처리."""
        indices, values = fitted_encoder.encode("")
        assert indices == []
        assert values == []

    def test_bm25_tf_normalization(self, encoder):
        """BM25 TF 정규화 동작 확인."""
        # 같은 단어가 반복되면 TF가 증가하지만, 정규화됨
        corpus = ["apple banana", "apple apple apple banana"]
        encoder.fit(corpus)

        # 단일 apple
        _, values1 = encoder.encode("apple")
        # 반복된 apple
        _, values2 = encoder.encode("apple apple apple")

        # 반복 시 점수가 증가하지만 선형적이지 않음 (saturation)
        assert values2[0] > values1[0]  # 반복으로 점수 증가
        # 하지만 3배는 아님 (BM25 saturation)
        assert values2[0] < values1[0] * 3

    def test_vocabulary_indices_are_unique(self, fitted_encoder):
        """vocabulary 인덱스가 고유해야 함."""
        indices = list(fitted_encoder._vocab.values())
        assert len(indices) == len(set(indices))

    def test_fit_with_empty_corpus(self, encoder):
        """빈 코퍼스로 fit 시 처리."""
        # 빈 리스트로 fit하면 에러 없이 처리
        encoder.fit([])
        assert encoder._avg_dl == 1.0  # 기본값
        assert len(encoder._vocab) == 0

    def test_custom_k1_b_parameters(self):
        """커스텀 k1, b 파라미터 적용 확인."""
        encoder1 = SparseEncoder(k1=1.0, b=0.5)
        encoder2 = SparseEncoder(k1=2.0, b=0.9)

        corpus = ["word " * 100]  # 긴 문서
        encoder1.fit(corpus)
        encoder2.fit(corpus)

        _, values1 = encoder1.encode("word")
        _, values2 = encoder2.encode("word")

        # 다른 파라미터로 다른 점수
        assert values1 != values2


class TestEmbedder:
    """Embedder (Dense) 테스트 - OpenAI API 모킹."""

    @pytest.fixture
    def embedder(self):
        """Embedder 인스턴스."""
        return Embedder()

    def test_generate_embeddings_single_chunk(self, embedder, sample_child_chunks, mock_openai_embeddings):
        """단일 청크 임베딩 생성 테스트."""
        chunks = sample_child_chunks[:1]
        embeddings = embedder.generate_embeddings(chunks)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536  # 더미 벡터 차원
        mock_openai_embeddings.assert_called_once()

    def test_generate_embeddings_multiple_chunks(self, embedder, sample_child_chunks, mock_openai_embeddings):
        """여러 청크 임베딩 생성 테스트."""
        embeddings = embedder.generate_embeddings(sample_child_chunks)

        assert len(embeddings) == len(sample_child_chunks)
        assert all(len(e) == 1536 for e in embeddings)

    def test_generate_embeddings_batching(self, embedder, mock_openai_embeddings):
        """배치 처리 테스트 (100개 초과 시 분할)."""
        from src.ingestion.processor import ChildChunk

        # 150개 청크 생성
        chunks = [
            ChildChunk(
                chunk_id=f"chunk-{i}",
                parent_id="parent-1",
                content=f"Content {i}",
                source="test",
                url="https://example.com",
                author="tester",
            )
            for i in range(150)
        ]

        embeddings = embedder.generate_embeddings(chunks)

        assert len(embeddings) == 150
        # 2번 호출됨 (100 + 50)
        assert mock_openai_embeddings.call_count == 2

    def test_generate_embeddings_empty_list(self, embedder, mock_openai_embeddings):
        """빈 리스트 처리."""
        embeddings = embedder.generate_embeddings([])

        assert embeddings == []
        mock_openai_embeddings.assert_not_called()
