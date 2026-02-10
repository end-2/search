import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

import tiktoken
from bs4 import BeautifulSoup

from src.common.config import get_settings
from src.ingestion.adapters.base import DocumentACL, RawDocument

logger = logging.getLogger(__name__)


@dataclass
class ParentChunk:
    """Parent Chunk: LLM 컨텍스트 제공용 큰 단위 텍스트."""

    chunk_id: str
    doc_id: str
    content: str
    source: str
    url: str
    author: str
    metadata: Dict = field(default_factory=dict)
    permissions: DocumentACL = field(default_factory=DocumentACL)


@dataclass
class ChildChunk:
    """Child Chunk: 벡터 검색용 작은 단위 텍스트."""

    chunk_id: str
    parent_id: str
    content: str
    source: str
    url: str
    author: str
    metadata: Dict = field(default_factory=dict)
    permissions: DocumentACL = field(default_factory=DocumentACL)


class TextCleaner:
    """텍스트 전처리: HTML 제거, 특수문자 정제."""

    @staticmethod
    def clean(text: str) -> str:
        # HTML 태그 제거
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        # 연속 공백/개행 정리
        text = re.sub(r"\s+", " ", text).strip()
        # 제어 문자 제거
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text


class Chunker:
    """
    Parent-Child 전략 기반 텍스트 분할기.
    - Parent: 큰 단위 (1000~2000 tokens) → 문맥 유지용
    - Child: 작은 단위 (200~400 tokens) → 검색 정밀도용
    """

    def __init__(self):
        settings = get_settings()
        self._parent_size = settings.parent_chunk_size
        self._parent_overlap = settings.parent_chunk_overlap
        self._child_size = settings.child_chunk_size
        self._child_overlap = settings.child_chunk_overlap
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _split_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """토큰 기준으로 텍스트를 분할합니다."""
        tokens = self._tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start = end - overlap
        return chunks

    def create_parent_child_chunks(
        self, document: RawDocument
    ) -> tuple[List[ParentChunk], List[ChildChunk]]:
        """
        하나의 문서를 Parent/Child 청크로 분할합니다.
        """
        parents: List[ParentChunk] = []
        children: List[ChildChunk] = []

        parent_texts = self._split_by_tokens(
            document.content, self._parent_size, self._parent_overlap
        )

        for parent_text in parent_texts:
            parent_id = str(uuid.uuid4())
            parent = ParentChunk(
                chunk_id=parent_id,
                doc_id=document.doc_id,
                content=parent_text,
                source=document.source,
                url=document.url,
                author=document.author,
                metadata=document.metadata,
                permissions=document.permissions,
            )
            parents.append(parent)

            child_texts = self._split_by_tokens(
                parent_text, self._child_size, self._child_overlap
            )
            for child_text in child_texts:
                child = ChildChunk(
                    chunk_id=str(uuid.uuid4()),
                    parent_id=parent_id,
                    content=child_text,
                    source=document.source,
                    url=document.url,
                    author=document.author,
                    metadata=document.metadata,
                    permissions=document.permissions,
                )
                children.append(child)

        return parents, children


class Processor:
    """Ingestion Processor: 문서 전처리 및 청킹을 담당합니다."""

    def __init__(self):
        self._cleaner = TextCleaner()
        self._chunker = Chunker()

    def process(
        self, documents: List[RawDocument]
    ) -> tuple[List[ParentChunk], List[ChildChunk]]:
        """
        원본 문서 리스트를 정제 후 Parent/Child 청크로 변환합니다.
        """
        all_parents: List[ParentChunk] = []
        all_children: List[ChildChunk] = []

        for doc in documents:
            doc.content = self._cleaner.clean(doc.content)
            if not doc.content:
                continue

            parents, children = self._chunker.create_parent_child_chunks(doc)
            all_parents.extend(parents)
            all_children.extend(children)

        logger.info(
            f"{len(documents)}개 문서 → Parent {len(all_parents)}개, Child {len(all_children)}개로 분할"
        )
        return all_parents, all_children
