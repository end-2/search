from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class DocumentACL:
    """문서의 접근 제어 목록(ACL)."""

    users: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    level: str = "private"  # "public" | "private"


@dataclass
class RawDocument:
    """어댑터에서 가져온 원본 문서."""

    doc_id: str
    title: str
    content: str
    source: str  # "slack" | "jira" | "wiki" | "notion"
    url: str = ""
    author: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    permissions: DocumentACL = field(default_factory=DocumentACL)


class DataSourceAdapter(ABC):
    """
    모든 데이터 소스 어댑터의 추상 기본 클래스입니다.
    Strategy Pattern을 적용하여 새로운 소스 추가 시 이 클래스만 구현하면 됩니다.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """데이터 소스 이름 (예: 'slack', 'jira')."""
        ...

    @abstractmethod
    def fetch_updates(self, last_sync_time: Optional[datetime] = None) -> List[RawDocument]:
        """
        마지막 동기화 시점 이후의 업데이트된 문서를 가져옵니다.

        Args:
            last_sync_time: 마지막 동기화 시점. None이면 전체 수집.
        Returns:
            RawDocument 리스트
        """
        ...

    @abstractmethod
    def extract_permissions(self, raw_data: dict) -> DocumentACL:
        """
        원본 데이터에서 접근 권한(ACL) 정보를 추출합니다.

        Args:
            raw_data: 소스 API로부터 받은 원시 데이터
        Returns:
            DocumentACL 객체
        """
        ...
