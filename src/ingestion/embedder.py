import logging
import math
from collections import Counter
from typing import Dict, List, Tuple

from src.common.config import get_settings
from src.common.openai_utils import create_embeddings
from src.ingestion.processor import ChildChunk

logger = logging.getLogger(__name__)


class Embedder:
    """OpenAI text-embedding 모델을 사용하여 Child Chunk를 Dense Vector로 변환합니다."""

    BATCH_SIZE = 100

    def generate_embeddings(self, chunks: List[ChildChunk]) -> List[List[float]]:
        """
        Child Chunk 리스트의 Dense Embedding을 생성합니다.
        API 제한을 고려하여 배치 단위로 처리합니다.
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            texts = [chunk.content for chunk in batch]
            embeddings = create_embeddings(texts)
            all_embeddings.extend(embeddings)

        logger.info(f"{len(chunks)}개 Child Chunk에 대한 Dense Embedding 생성 완료")
        return all_embeddings


class SparseEncoder:
    """
    BM25 기반 Sparse Vector 생성기.
    Hybrid Search를 위해 Child Chunk의 키워드 기반 Sparse Vector를 생성합니다.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._vocab: Dict[str, int] = {}

    def fit(self, texts: List[str]):
        """IDF 및 평균 문서 길이를 계산합니다."""
        n = len(texts)
        df: Counter = Counter()
        total_len = 0

        for text in texts:
            tokens = self._tokenize(text)
            total_len += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        self._avg_dl = total_len / n if n > 0 else 1.0

        # Vocabulary 구축 (인덱스 매핑)
        self._vocab = {term: idx for idx, term in enumerate(sorted(df.keys()))}

        # IDF 계산
        for term, freq in df.items():
            self._idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """
        텍스트를 BM25 Sparse Vector로 변환합니다.
        Returns:
            (indices, values) 형태의 Sparse Vector
        """
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        dl = len(tokens)

        indices: List[int] = []
        values: List[float] = []

        for term, freq in tf.items():
            if term not in self._vocab:
                continue
            idf = self._idf.get(term, 0.0)
            tf_norm = (freq * (self._k1 + 1)) / (
                freq + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
            )
            score = idf * tf_norm
            if score > 0:
                indices.append(self._vocab[term])
                values.append(score)

        return indices, values

    def encode_batch(self, chunks: List[ChildChunk]) -> List[Tuple[List[int], List[float]]]:
        """Child Chunk 리스트에 대한 Sparse Vector를 일괄 생성합니다."""
        results = [self.encode(chunk.content) for chunk in chunks]
        logger.info(f"{len(chunks)}개 Child Chunk에 대한 Sparse Vector 생성 완료")
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """간단한 공백 기반 토큰화."""
        return [w.lower().strip(".,!?;:\"'()[]{}") for w in text.split() if w.strip()]
